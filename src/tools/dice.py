"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Training and evaluation codes for 
3D human body mesh reconstruction from an image
"""

from __future__ import absolute_import, division, print_function
import argparse
import os
import os.path as op
import code
import json
import time
import datetime
import torch
import torchvision.models as models
from torchvision.utils import make_grid
import numpy as np
import cv2
from src.modeling.bert import BertConfig
from src.modeling.bert.model import DICE_Module
from src.modeling.bert.model import DICE_Network
from src.modeling._mano import MANO, Mesh
from src.modeling.hrnet.hrnet_cls_net_featmaps import get_cls_net
from src.modeling.hrnet.config import config as hrnet_config
from src.modeling.hrnet.config import update_config as hrnet_update_config
import src.modeling.data.config as cfg
from src.datasets.build import make_decaf_data_loader, make_decaf_and_inthewild_data_loader

from src.utils.logger import setup_logger
from src.utils.comm import synchronize, is_main_process, get_rank, get_world_size, all_gather
from src.utils.miscellaneous import mkdir, set_seed
from src.utils.metric_logger import AverageMeter, EvalMetricsLogger
from src.utils.renderer import Renderer, visualize_reconstruction, visualize_reconstruction_test
from src.utils.metric_pampjpe import reconstruction_error
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

from src.decaf.FLAME_util import FLAME, get_FLAME_faces, flame_forwarding
from src.decaf.util import denormalize_keys, keypoint_overlay
from src.decaf.tracking_util import mano_forwarding
import mano
import copy
from src.utils.plausibility_eval_simple import get_plausibility_metrics, collision_detection_sdf
from src.utils.chamfer_distance import ChamferDistance
import random
from collections import OrderedDict
from pytorch3d.renderer import MeshRasterizer, FoVPerspectiveCameras, RasterizationSettings, PerspectiveCameras, OrthographicCameras, PointLights, SoftPhongShader, TexturesVertex
from pytorch3d.structures import Meshes
from matplotlib import pyplot as plt
from src.utils.visualizer import visualize_keypoints_single
import matplotlib.cm as cm


def save_checkpoint(model, args, epoch, iteration, num_trial=10, face_d_model=None, hand_d_model=None):
    checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}-{}'.format(
        epoch, iteration))
    if not is_main_process():
        return checkpoint_dir
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    for i in range(num_trial):
        try:
            torch.save(model_to_save, op.join(checkpoint_dir, 'model.bin'))
            torch.save(model_to_save.state_dict(), op.join(checkpoint_dir, 'state_dict.bin'))
            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            if face_d_model is not None:
                torch.save(face_d_model, op.join(checkpoint_dir, 'face_d_model.bin'))
                torch.save(face_d_model.state_dict(), op.join(checkpoint_dir, 'face_d_state_dict.bin'))
            if hand_d_model is not None:
                torch.save(hand_d_model, op.join(checkpoint_dir, 'hand_d_model.bin'))
                torch.save(hand_d_model.state_dict(), op.join(checkpoint_dir, 'hand_d_state_dict.bin'))
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass
    else:
        logger.info("Failed to save checkpoint after {} trails.".format(num_trial))
    return checkpoint_dir

def save_scores(args, split, mpjpe, pampjpe, mpve):
    eval_log = []
    res = {}
    res['mPJPE'] = mpjpe
    res['PAmPJPE'] = pampjpe
    res['mPVE'] = mpve
    eval_log.append(res)
    with open(op.join(args.output_dir, split+'_eval_logs.json'), 'w') as f:
        json.dump(eval_log, f)
    logger.info("Save eval scores to {}".format(args.output_dir))
    return

def adjust_learning_rate(optimizer, epoch, args):
    """
    Sets the learning rate to the initial LR decayed by x every y epochs
    x = 0.1, y = args.num_train_epochs/2.0 = 100
    """
    lr = args.lr * (0.1 ** (epoch // (args.num_train_epochs/2.0)  ))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def mean_per_joint_position_error(pred, gt):
    """ 
    Compute mPJPE
    """

    with torch.no_grad():
        error = torch.sqrt( ((pred - gt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        return error

def mean_per_vertex_error(pred, gt):
    """
    Compute mPVE
    """
    with torch.no_grad():
        error = torch.sqrt( ((pred - gt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        return error

def deform_error(pred, gt):
    """
    Compute mPVE
    """
    # B, N, 3
    large_deform_error = None
    batch_size = gt.shape[0]
    for i in range(batch_size):
        large_deform_indices = torch.norm(gt[i], p=2, dim=1) > 0.005
        pred_large_deform = pred[i][large_deform_indices] # shape: N, 3
        gt_large_deform = gt[i][large_deform_indices] # shape: N, 3
        if pred_large_deform.shape[0] > 0:
            error = torch.sqrt( ((pred_large_deform - gt_large_deform) ** 2) ).sum(dim=-1).cpu().numpy()
            if large_deform_error is None:
                large_deform_error = error
            else:
                large_deform_error = np.concatenate((large_deform_error, error), axis=0)
    with torch.no_grad():
        deform_error = torch.sqrt( ((pred - gt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

    if large_deform_error is None:
        large_deform_error = []

    return deform_error, large_deform_error


def keypoint_2d_loss(criterion_keypoints, pred_keypoints_2d, gt_keypoints_2d, has_2d_kp, image_size, device):
    """
    Compute 2D reprojection loss if 2D keypoint annotations are available.
    The confidence (conf) is binary and indicates whether the keypoints exist or not.
    """
    pred_keypoints_2d = pred_keypoints_2d / image_size
    pred_keypoints_2d = pred_keypoints_2d[has_2d_kp == 1]
    gt_keypoints_2d = gt_keypoints_2d[has_2d_kp == 1]
    if len(gt_keypoints_2d) > 0:
        loss = criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d).mean()
    else:
        loss = torch.FloatTensor(1).fill_(0.).to(device)
    return loss

def keypoint_3d_loss(criterion_keypoints, pred_keypoints_3d, gt_keypoints_3d, has_3d_kp, device):
    """
    Compute 3D keypoint loss if 3D keypoint annotations are available.
    """
    # gt keypoints have already been normalized by their center.
    pred_keypoints_3d = pred_keypoints_3d[has_3d_kp == 1]
    gt_keypoints_3d = gt_keypoints_3d[has_3d_kp == 1]
    if len(gt_keypoints_3d) > 0:
        return criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)
    else:
        return torch.FloatTensor(1).fill_(0.).to(device)


def vertices_loss(criterion_vertices, pred_vertices, gt_vertices, has_3d_mesh, device):
    """
    Compute per-vertex loss if vertex annotations are available.
    """
    pred_vertices = pred_vertices[has_3d_mesh == 1]
    gt_vertices = gt_vertices[has_3d_mesh == 1]
    if len(gt_vertices) > 0:
        return criterion_vertices(pred_vertices, gt_vertices)
    else:
        return torch.FloatTensor(1).fill_(0.).to(device)


def params_loss(criterion_params, pred_params, gt_params, has_params, device):
    pred_params = pred_params[has_params == 1]
    gt_params = gt_params[has_params == 1]
    if len(gt_params) > 0:
        return criterion_params(pred_params, gt_params)
    else:
        return torch.FloatTensor(1).fill_(0.).to(device)


def contact_loss(criterion_contact, pred_contact, gt_contact, has_contact, device):
    """
    Compute contact loss if contact annotations are available.
    """
    pred_contact = pred_contact[has_contact == 1]
    gt_contact = gt_contact[has_contact == 1]
    pred_contact = torch.clamp(pred_contact, 0.00001, 0.99999)
    gt_contact = torch.clamp(gt_contact, 0, 1)
    
    if contains_nan(pred_contact):
        print(pred_contact.shape)
        print("pred_contact contains nan")
        return torch.FloatTensor(1).fill_(0.).to(device)
    if contains_nan(gt_contact):
        print("gt_contact contains nan")
        return torch.FloatTensor(1).fill_(0.).to(device)

    if len(gt_contact) > 0:
        return criterion_contact(pred_contact.float(), gt_contact.float())
    else:
        return torch.FloatTensor(1).fill_(0.).to(device)
    

def deform_loss(criterion_deform, pred_def, gt_def, w_reg, deform_weight_mask, has_deform, device):
    pred_def = pred_def[has_deform == 1]
    gt_def = gt_def[has_deform == 1]
    deform_weight_mask = deform_weight_mask[has_deform == 1]
    if len(gt_def) > 0:
        return criterion_deform(pred_def, gt_def, w_reg, deform_weight_mask)
    else:
        return torch.FloatTensor(1).fill_(0.).to(device), torch.FloatTensor(1).fill_(0.).to(device), torch.FloatTensor(1).fill_(0.).to(device)

def deform_reg_loss(criterion_deform_reg, pred_def, gt_def, w_reg, deform_weight_mask, has_deform_reg, device):
    pred_def = pred_def[has_deform_reg == 1]
    gt_def = gt_def[has_deform_reg == 1]
    deform_weight_mask = deform_weight_mask[has_deform_reg == 1]
    if len(gt_def) > 0:
        return criterion_deform_reg(pred_def, gt_def, w_reg, deform_weight_mask)
    else:
        return torch.FloatTensor(1).fill_(0.).to(device)

def contains_nan(tensor):
    return torch.isnan(tensor).any().item()


def presence_loss(criterion_presence, pred_presence, gt_presence, has_presence, device):
    pred_presence = pred_presence[has_presence == 1]
    gt_presence = gt_presence[has_presence == 1]
    pred_presence = torch.clamp(pred_presence, 0.00001, 0.99999)
    gt_presence = torch.clamp(gt_presence, 0, 1)

    if contains_nan(pred_presence):
        print("pred_presence contains nan")
        return torch.FloatTensor(1).fill_(0.).to(device)
    if contains_nan(gt_presence):
        print("gt_presence contains nan")
        return torch.FloatTensor(1).fill_(0.).to(device)

    if len(gt_presence) > 0:
        return criterion_presence(pred_presence.float(), gt_presence.float())
    else:
        return torch.FloatTensor(1).fill_(0.).to(device)

def rectify_pose(pose):
    pose = pose.copy()
    R_mod = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
    R_root = cv2.Rodrigues(pose[:3])[0]
    new_root = R_root.dot(R_mod)
    pose[:3] = cv2.Rodrigues(new_root)[0].reshape(3)
    return pose

def normal_vector_loss(coord_out, coord_gt, face):
    if len(coord_gt) > 0:
        face = torch.LongTensor(face).cuda()

        v1_out = coord_out[:,face[:,1],:] - coord_out[:,face[:,0],:]
        v1_out = F.normalize(v1_out, p=2, dim=2) # L2 normalize to make unit vector
        v2_out = coord_out[:,face[:,2],:] - coord_out[:,face[:,0],:]
        v2_out = F.normalize(v2_out, p=2, dim=2) # L2 normalize to make unit vector
        v3_out = coord_out[:,face[:,2],:] - coord_out[:,face[:,1],:]
        v3_out = F.normalize(v3_out, p=2, dim=2) # L2 nroamlize to make unit vector

        v1_gt = coord_gt[:,face[:,1],:] - coord_gt[:,face[:,0],:]
        v1_gt = F.normalize(v1_gt, p=2, dim=2) # L2 normalize to make unit vector
        v2_gt = coord_gt[:,face[:,2],:] - coord_gt[:,face[:,0],:]
        v2_gt = F.normalize(v2_gt, p=2, dim=2) # L2 normalize to make unit vector
        normal_gt = torch.cross(v1_gt, v2_gt, dim=2)
        normal_gt = F.normalize(normal_gt, p=2, dim=2) # L2 normalize to make unit vector


        cos1 = torch.abs(torch.sum(v1_out * normal_gt, 2, keepdim=True))
        cos2 = torch.abs(torch.sum(v2_out * normal_gt, 2, keepdim=True))
        cos3 = torch.abs(torch.sum(v3_out * normal_gt, 2, keepdim=True))

        loss = torch.cat((cos1, cos2, cos3),1)
        loss = loss.mean()
        return loss

def normal_vector_loss_cosine(coord_out, coord_gt, face):
    if len(coord_gt) > 0:
        face = torch.LongTensor(face).cuda()

        v1_out = coord_out[:,face[:,1],:] - coord_out[:,face[:,0],:]
        v1_out = F.normalize(v1_out, p=2, dim=2) # L2 normalize to make unit vector
        v2_out = coord_out[:,face[:,2],:] - coord_out[:,face[:,0],:]
        v2_out = F.normalize(v2_out, p=2, dim=2) # L2 normalize to make unit vector
        normal_pred = torch.cross(v1_out, v2_out, dim=2)
        normal_pred = F.normalize(normal_pred, p=2, dim=2) # L2 normalize to make unit vector

        v1_gt = coord_gt[:,face[:,1],:] - coord_gt[:,face[:,0],:]
        v1_gt = F.normalize(v1_gt, p=2, dim=2) # L2 normalize to make unit vector
        v2_gt = coord_gt[:,face[:,2],:] - coord_gt[:,face[:,0],:]
        v2_gt = F.normalize(v2_gt, p=2, dim=2) # L2 normalize to make unit vector
        normal_gt = torch.cross(v1_gt, v2_gt, dim=2)
        normal_gt = F.normalize(normal_gt, p=2, dim=2) # L2 normalize to make unit vector

        cos = normal_pred * normal_gt
        cos = torch.sum(cos, dim=-1, keepdim=True)

        loss = 1 - cos
        loss = loss.mean()
        return loss

def edge_length_loss(coord_out, coord_gt, face):
    if len(coord_gt) > 0:
        face = torch.LongTensor(face).cuda()

        d1_out = torch.sqrt(torch.sum((coord_out[:,face[:,0],:] - coord_out[:,face[:,1],:])**2,2,keepdim=True) + 1e-6)
        d2_out = torch.sqrt(torch.sum((coord_out[:,face[:,0],:] - coord_out[:,face[:,2],:])**2,2,keepdim=True) + 1e-6)
        d3_out = torch.sqrt(torch.sum((coord_out[:,face[:,1],:] - coord_out[:,face[:,2],:])**2,2,keepdim=True) + 1e-6)

        d1_gt = torch.sqrt(torch.sum((coord_gt[:,face[:,0],:] - coord_gt[:,face[:,1],:])**2,2,keepdim=True) + 1e-6)
        d2_gt = torch.sqrt(torch.sum((coord_gt[:,face[:,0],:] - coord_gt[:,face[:,2],:])**2,2,keepdim=True) + 1e-6)
        d3_gt = torch.sqrt(torch.sum((coord_gt[:,face[:,1],:] - coord_gt[:,face[:,2],:])**2,2,keepdim=True) + 1e-6)

        diff1 = torch.abs(d1_out - d1_gt)
        diff2 = torch.abs(d2_out - d2_gt)
        diff3 = torch.abs(d3_out - d3_gt)

        loss = torch.cat((diff1, diff2, diff3), 1)
        loss = loss.mean()
        return loss

def save_mesh(path, V, F):
    with open(path, "w") as f:
        for i in range(V.shape[0]):
            f.write("v %f %f %f\n" % (V[i, 0], V[i, 1], V[i, 2]))
        for j in range(F.shape[0]):
            f.write("f %d %d %d\n" % (1+F[j, 0], 1+F[j, 1], 1+F[j, 2]))

def visualize_keypoints(img, keypoints, color=(255,0,0), w=224, h=224):
    keypoints_vis = copy.deepcopy(keypoints)
    keypoints_vis = denormalize_keys(keys=keypoints_vis,w=w,h=h)
    keypoints_vis = keypoints_vis.squeeze().cpu().detach().numpy()
    img = img.squeeze().cpu().permute(1,2,0).detach().numpy()
    img = 255 * (img - img.min()) / (img.max() - img.min())
    img = np.ascontiguousarray(img, dtype=np.uint8)
    img = keypoint_overlay(
        keypoints_vis.astype(int),
        c=color,
        img=img)
    return img

def criterion_deformation(pred_def, gt_def, w_reg, deform_weight_mask):
    b = pred_def.shape[0]
    weight_mask = torch.ones_like(gt_def) + torch.norm(gt_def, p=2, dim=2, keepdim=True) * 5000

    large_def_mask = w_reg * (torch.norm(pred_def, p=2, dim=2, keepdim=True) > 0.03).float()

    loss_deviation = ( (pred_def - gt_def) ** 2) * weight_mask
    loss_deviation = loss_deviation.sum(dim=-1).mean(dim=-1) # B, N, 3 -> B
    loss_deviation = 100 * loss_deviation.mean()

    loss_reg = large_def_mask * torch.abs(pred_def)
    loss_reg = loss_reg.sum(dim=-1).mean(dim=-1) # B, N, 3 -> B
    loss_reg = loss_reg.mean()

    loss = loss_deviation + loss_reg
    return loss, loss_deviation, loss_reg

def criterion_deformation_reg(pred_def, gt_def, w_reg, deform_weight_mask):
    loss_reg = torch.abs(pred_def)
    loss_reg = loss_reg.sum(dim=-1).mean(dim=-1) # B, N, 3 -> B
    loss_reg = loss_reg.mean()
    return loss_reg

def criterion_touch(chamfer_distance, pred_face_vs, pred_hand_vs, pred_face_contacts, pred_hand_contacts):
    batch_size = pred_face_vs.shape[0]
    loss = torch.tensor(0).to(pred_face_vs.device)
    for i in range(batch_size):
        face_contact_mask = (pred_face_contacts[i] > 0.5).squeeze()
        hand_contact_mask = (pred_hand_contacts[i] > 0.5).squeeze()
        pred_face_contact_vs = pred_face_vs[i][face_contact_mask].unsqueeze(0)
        pred_hand_contact_vs = pred_hand_vs[i][hand_contact_mask].unsqueeze(0)
        if 0 in pred_face_contact_vs.shape or 0 in pred_hand_contact_vs.shape:
            continue

        cham_x, cham_y, idx_x, idx_y = chamfer_distance(pred_face_contact_vs, pred_hand_contact_vs)
        
        loss = loss + cham_x.mean() + cham_y.mean()
    loss = loss / batch_size
    return loss

def criterion_collision(chamfer_distance, pred_face_vs, pred_hand_vs, flame_faces):
    batch_size = pred_face_vs.shape[0]
    batch_col_dist = torch.tensor(0).to(pred_face_vs.device)
    for i in range(batch_size):
        collision_flag , _, _, \
        _,_,_ = \
        collision_detection_sdf(obj_faces=flame_faces,
                                        obj_vs=pred_face_vs[i].cpu().detach().numpy() ,
                                        query_vs=pred_hand_vs[i].cpu().detach().numpy())

        pred_hand_collision_vs = pred_hand_vs[i][collision_flag].unsqueeze(0)
        if 0 in pred_hand_collision_vs.shape:
            continue

        _, cham_y, _, _ = chamfer_distance(pred_face_vs[i].unsqueeze(0), pred_hand_collision_vs)

        batch_col_dist = batch_col_dist + cham_y.sum() / pred_face_vs.shape[1]
    batch_col_dist = batch_col_dist / batch_size
    return batch_col_dist


def get_rotation_matrix(s, t):
    R = torch.zeros((s.shape[0], 3, 3)).to(s.device)
    R[:, 0, 0] = s
    R[:, 1, 1] = s
    R[:, 2, 2] = 1
    R[:, 0, 2] = t[:, 0]
    R[:, 1, 2] = t[:, 1]
    return R

def merge_faces(hand_faces, head_faces):
    hand_faces_clone = copy.deepcopy(hand_faces)
    head_faces_clone = copy.deepcopy(head_faces)
    head_faces_clone = head_faces_clone + hand_faces_clone.max() + 1

    return torch.cat([hand_faces_clone, head_faces_clone], dim=0)

def merge_verts(hand_verts, face_verts):
    return torch.cat([hand_verts, face_verts], dim=1)

def depth_to_heatmap(depth_map):
    eps = 1e-6
    # Normalize depth map values between 0 and 1
    normalized_depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map) + eps)
    
    # Apply colormap (jet in this case, you can change it)
    heatmap = cm.jet(normalized_depth_map)
    
    return heatmap[:, :, :3] # excluding alpha channel

def depth_to_heatmap_batch(depth_map):
    batch_size = depth_map.shape[0]
    heatmaps = []
    for i in range(batch_size):
        heatmap = depth_to_heatmap(depth_map[i])
        heatmaps.append(heatmap)
    return heatmaps

def process_zbuf(zbuf):
    # set background to be 2m
    zbuf[zbuf == -1] = 2.0
    return zbuf


# Main loss function used for ZoeDepth. Copy/paste from AdaBins repo (https://github.com/shariqfarooq123/AdaBins/blob/0952d91e9e762be310bb4cd055cbfe2448c0ce20/loss.py#L7)
class SILogLoss(nn.Module):
    """SILog loss (pixel-wise)"""
    def __init__(self, beta=0.15):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'
        self.beta = beta

    def forward(self, input, target, mask=None, interpolate=True, return_interpolated=False):
        alpha = 1e-7
        g = torch.log(input + alpha) - torch.log(target + alpha)
        Dg = torch.var(g)

        loss = torch.sqrt(Dg)

        if torch.isnan(loss):
            print("Nan SILog loss")
            print("input:", input.shape)
            print("target:", target.shape)
            print("G", torch.sum(torch.isnan(g)))
            if input.numel() > 0:
                print("Input min max", torch.min(input), torch.max(input))
                print("Target min max", torch.min(target), torch.max(target))
            print("Dg", torch.isnan(Dg))
            print("loss", torch.isnan(loss))
            return torch.FloatTensor(1).fill_(0.).to(input.device)

        if not return_interpolated:
            return loss

        return loss, None


def depth_loss(criterion_depth, pred_depth, gt_depth, mask, has_depth, device):
    pred_depth = pred_depth[has_depth == 1]
    gt_depth = gt_depth[has_depth == 1]
    mask = mask[has_depth == 1]
    if len(gt_depth) > 0:
        return criterion_depth(pred_depth, gt_depth, mask)
    else:
        return torch.FloatTensor(1).fill_(0.).to(device)

def depth_loss_kp(criterion_depth, pred_depth, gt_depth, pred_indices, gt_indices, has_depth, device):
    pred_depth = pred_depth[has_depth == 1]
    gt_depth = gt_depth[has_depth == 1]

    if len(gt_depth) > 0:
        pred_indices = torch.clamp(pred_indices[has_depth == 1], 0, pred_depth.shape[2] - 1)
        gt_indices = torch.clamp(gt_indices[has_depth == 1], 0, gt_depth.shape[2] - 1)

        has_depth_count = pred_depth.shape[0]
        pred_indices = pred_indices[:, :, 0] * pred_depth.shape[2] + pred_indices[:, :, 1]
        gt_indices = gt_indices[:, :, 0] * gt_depth.shape[2] + gt_indices[:, :, 1]
        pred_depth_kps = torch.gather(pred_depth.view(has_depth_count, -1), 1, pred_indices)
        gt_depth_kps = torch.gather(gt_depth.view(has_depth_count, -1), 1, gt_indices)

        foreground_indices = (pred_depth_kps > 0)
        pred_depth_kps = pred_depth_kps[foreground_indices]
        gt_depth_kps = gt_depth_kps[foreground_indices]
        return criterion_depth(pred_depth_kps, gt_depth_kps)
    else:
        return torch.FloatTensor(1).fill_(0.).to(device)

def discriminator_loss(criterion_discriminator, pred_prob, gt_prob, has_discr, device):
    pred_prob = pred_prob[has_discr == 1]
    gt_prob = gt_prob[has_discr == 1]
    pred_prob = torch.clamp(pred_prob, 0.00001, 0.99999)
    gt_prob = torch.clamp(gt_prob, 0, 1)

    if contains_nan(pred_prob):
        print("pred_prob contains nan")
        return torch.FloatTensor(1).fill_(0.).to(device)
    if contains_nan(gt_prob):
        print("gt_prob contains nan")
        return torch.FloatTensor(1).fill_(0.).to(device)

    if len(gt_prob) > 0:
        return criterion_discriminator(pred_prob, gt_prob)
    else:
        return torch.FloatTensor(1).fill_(0.).to(device)

def set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


def save_mesh_to_ply(vertices, faces, texture, filename='mesh.ply'):
    """
    Save a mesh to a PLY file.

    Parameters:
    vertices (torch.Tensor): Tensor of shape (N, 3) containing vertex coordinates.
    faces (torch.Tensor): Tensor of shape (F, 3) containing face indices.
    texture (torch.Tensor): Tensor of shape (N, 3) containing vertex texture colors.
    filename (str): The filename for the PLY file.
    """

    # Convert tensors to numpy arrays
    vertices_np = vertices.numpy()
    faces_np = faces.numpy()
    texture_np = texture.numpy()

    # Combine vertices and texture into a single array
    vertices_with_texture = np.hstack([vertices_np, texture_np])

    # Define the vertex and face elements
    vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'f4'), ('green', 'f4'), ('blue', 'f4')]
    face_dtype = [('vertex_indices', 'i4', (3,))]

    vertex_data = np.array([tuple(v) for v in vertices_with_texture], dtype=vertex_dtype)
    face_data = np.array([([f[0], f[1], f[2]],) for f in faces_np], dtype=face_dtype)

    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    face_element = PlyElement.describe(face_data, 'face')

    # Write to a PLY file
    ply_data = PlyData([vertex_element, face_element])
    ply_data.write(filename)
    print(f"saved ply to {filename}")


class Hand_Discriminator(nn.Module):
    def __init__(self):
        super(Hand_Discriminator, self).__init__()
        self.model = torch.nn.Sequential(
            nn.Linear(45+10, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, params):
        return self.model(params)

class Face_Discriminator(nn.Module):
    def __init__(self):
        super(Face_Discriminator, self).__init__()
        self.model = torch.nn.Sequential(
            nn.Linear(100+50+3, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, params):
        return self.model(params)


from PIL import Image, ImageDraw, ImageFont
import numpy as np

def overlay_number(image_array, text):
    # Create a blank image
    image = Image.fromarray(image_array)

    # Font settings
    font_size = 20
    font = ImageFont.load_default()
    font_color = (255, 255, 255)  # white color

    # Draw text on the image
    draw = ImageDraw.Draw(image)
    x_position = 0
    y_position = 0
    draw.text((x_position, y_position), text, font=font, fill=font_color)

    # Convert back to numpy array
    overlayed_image_array = np.array(image)

    return overlayed_image_array


def run(args, train_dataloader, val_dataloader, DICE_model, mano_model=None, hand_model=None, head_model=None, hand_sampler=None, head_sampler=None, hand_renderer=None, face_renderer=None, hand_F=None, head_F=None, hand_discriminator=None, face_discriminator=None):
    hand_model.eval()
    head_model.eval()
    mano_model.eval()

    hand_model.to(args.device)
    head_model.to(args.device)

    max_iter = len(train_dataloader)
    iters_per_epoch = max_iter // args.num_train_epochs

    optimizer = torch.optim.AdamW(params=list(DICE_model.parameters()),
                                           lr=args.lr,
                                           betas=(0.9, 0.999),
                                           weight_decay=1e-4)

    # define loss function (criterion) and optimizer
    criterion_2d_keypoints = torch.nn.L1Loss().cuda(args.device)
    criterion_keypoints = torch.nn.L1Loss().cuda(args.device)
    criterion_vertices = torch.nn.L1Loss().cuda(args.device)
    criterion_contact = torch.nn.BCELoss().cuda(args.device)
    criterion_params = torch.nn.L1Loss().cuda(args.device)
    criterion_presence = torch.nn.BCELoss().cuda(args.device)
    chamfer_distance = ChamferDistance()
    criterion_discriminator = torch.nn.BCELoss().cuda(args.device)

    if args.distributed:  
        DICE_model = torch.nn.parallel.DistributedDataParallel(
            DICE_model, device_ids=[args.local_rank], 
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

        logger.info(
                ' '.join(
                ['Local rank: {o}', 'Max iteration: {a}', 'iters_per_epoch: {b}','num_train_epochs: {c}',]
                ).format(o=args.local_rank, a=max_iter, b=iters_per_epoch, c=args.num_train_epochs)
            )

    start_training_time = time.time()
    end = time.time()
    DICE_model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    log_losses = AverageMeter()
    log_loss_2djoints_np = AverageMeter()
    log_loss_2djoints_p = AverageMeter()
    log_loss_3djoints_np = AverageMeter()
    log_loss_3djoints_p = AverageMeter()
    log_loss_vertices_np = AverageMeter()
    log_loss_vertices_p = AverageMeter()
    log_loss_self_sup = AverageMeter()
    log_eval_metrics = EvalMetricsLogger()
    log_loss_normal_vector = AverageMeter()
    log_loss_edge_length = AverageMeter()
    log_loss_contacts = AverageMeter()
    log_loss_params = AverageMeter()
    log_loss_deforms = AverageMeter()
    log_loss_collision = AverageMeter()
    log_loss_touch = AverageMeter()
    log_loss_deform_deviation = AverageMeter()
    log_loss_deform_reg = AverageMeter()
    log_loss_contact_presence = AverageMeter()

    # save mesh
    asset_path = args.data_path+"/assets/"
    flame_model_path = asset_path+"/generic_model.pkl"
    flame_faces = get_FLAME_faces(flame_model_path)

    stiffness = np.load("src/modeling/data/stiffness_final.npy")
    stiffness = torch.from_numpy(stiffness).float().cuda(args.device)
    criterion_depth = SILogLoss()


    hand_D_optimizer = torch.optim.AdamW(params=list(hand_discriminator.parameters()),
                                           lr=args.lr,
                                           betas=(0.9, 0.999),
                                           weight_decay=1e-4)
    
    face_D_optimizer = torch.optim.AdamW(params=list(face_discriminator.parameters()),
                                            lr=args.lr,
                                            betas=(0.9, 0.999),
                                            weight_decay=1e-4)
    

    for iteration, data in tqdm(enumerate(train_dataloader)):

        DICE_model.train()
        iteration += 1
        epoch = iteration // iters_per_epoch
        batch_size = data["single_img_seqs"].shape[0]
        adjust_learning_rate(optimizer, epoch, args)
        data_time.update(time.time() - end)

        stiffness_batch = stiffness.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)
        deform_weight_mask = torch.reciprocal((1 - stiffness_batch + 0.1) ** 1.5)
        deform_weight_mask = deform_weight_mask / deform_weight_mask.min()

        single_image = data["single_img_seqs"].cuda(args.device)
        if contains_nan(single_image):
            print("single_image contains nan")
            print(data["data_index"].cpu().tolist())
            raise ValueError("single_image contains nan")

        has_2d_kp = data["has_2d_kp"].cuda(args.device)
        has_3d_kp = data["has_3d_kp"].cuda(args.device)
        has_3d_mesh = data["has_3d_mesh"].cuda(args.device)
        has_depth = data["has_depth"].cuda(args.device)
        has_contact = data["has_contact"].cuda(args.device)
        has_params = data["has_params"].cuda(args.device)
        has_deform = data["has_deform"].cuda(args.device)
        has_deform_reg = (has_deform == 0)
        has_presence = has_contact
        has_discr = torch.ones_like(has_2d_kp) # discriminator applies to all data

        gt_depth_maps = data["depth_map"].cuda(args.device)

        data_index = data["data_index"].cpu().tolist()

        rh_ref_vs = data["rh_ref_vs"].cuda(args.device)
        head_ref_vs = data["head_ref_vs"].cuda(args.device)

        gt_hand_contact = data["rh_con_labels"].cuda(args.device).unsqueeze(-1)
        gt_head_contact = data["head_con_labels"].cuda(args.device).unsqueeze(-1)

        gt_hand_vertices = data["rh_vs_in_cam"].cuda(args.device)
        gt_head_vertices_undeformed = data["head_vs_in_cam"].cuda(args.device)
        gt_hand_keypoints = data["rh_keys_in_cam"].cuda(args.device)
        gt_head_keypoints = data["head_keys_in_cam"].cuda(args.device)
        gt_head_vertices_deformed = data["head_vs_in_cam_deformed"].cuda(args.device)
        gt_deformation = data["deformation_cam"].cuda(args.device)

        gt_head_center = gt_head_vertices_deformed.mean(dim=1)

        gt_head_vertices_undeformed = gt_head_vertices_undeformed - gt_head_center[:, None, :]
        gt_head_vertices_deformed = gt_head_vertices_deformed - gt_head_center[:, None, :]
        gt_hand_vertices = gt_hand_vertices - gt_head_center[:, None, :]
        gt_hand_keypoints = gt_hand_keypoints - gt_head_center[:, None, :]
        gt_head_keypoints = gt_head_keypoints - gt_head_center[:, None, :]
        
        head_vs_proj_single = data["head_vs_proj_single"].cuda(args.device)
        rh_vs_proj_single = data["rh_vs_proj_single"].cuda(args.device)
        head_keys_proj_single = data["head_keys_proj_single"].cuda(args.device)
        rh_keys_proj_single = data["rh_keys_proj_single"].cuda(args.device)

        head_vs_proj_single_sub2 = head_sampler.downsample(head_vs_proj_single.double())
        rh_vs_proj_single_sub = hand_sampler.downsample(rh_vs_proj_single.double())

        gt_head_vertices_sub2 = head_sampler.downsample(gt_head_vertices_undeformed.double()).float()
        gt_head_vertices_sub = head_sampler.downsample(gt_head_vertices_undeformed.double(), n1=0, n2=1).float()
        gt_hand_vertices_sub = hand_sampler.downsample(gt_hand_vertices.double()).float()

        gt_head_contact_sub2 = head_sampler.downsample(gt_head_contact.double()).float()
        gt_head_contact_sub = head_sampler.downsample(gt_head_contact.double(), n1=0, n2=1).float()
        gt_hand_contact_sub = hand_sampler.downsample(gt_hand_contact.double()).float()

        head_ref_vs_sub2 = head_sampler.downsample(head_ref_vs.double()).float()
        rh_ref_vs_sub = hand_sampler.downsample(rh_ref_vs.double()).float()
        head_ref_kps = head_model.convert_vs2landmarks(head_ref_vs)
        rh_ref_kps = mano_model.get_3d_joints(rh_ref_vs)

        mvm_percent = 0.3
        mvm_mask = np.ones((21+68+195+559,1))
        num_vertices = 21+68+195+559
        pb = np.random.random_sample()
        masked_num = int(pb * mvm_percent * num_vertices) # at most x% of the vertices could be masked
        indices = np.random.choice(np.arange(num_vertices),replace=False,size=masked_num)
        mvm_mask[indices,:] = 0.0
        mvm_mask = torch.from_numpy(mvm_mask).float()

        mvm_mask_ = mvm_mask.expand(batch_size,-1,2051)

        gt_rh_betas = data["rh_betas"].cuda(args.device)
        gt_rh_transl = data["rh_transl"].cuda(args.device)
        gt_rh_rot = data["rh_rot"].cuda(args.device)
        gt_rh_pose = data["rh_pose"].cuda(args.device)
        gt_face_shape = data["face_shape"].cuda(args.device)
        gt_face_exp = data["face_exp"].cuda(args.device)
        gt_face_pose = data["face_pose"].cuda(args.device)
        gt_face_rot = data["face_rot"].cuda(args.device)
        gt_face_transl = data["face_transl"].cuda(args.device)

        gt_contact_presence, _ = torch.max(torch.cat([gt_hand_contact, gt_head_contact], dim=1), dim=1)

        real_face_shape = data['sampled_face_shape'].cuda(args.device) # 100
        real_face_exp = data['sampled_face_exp'].cuda(args.device) # 50
        real_face_pose = data['sampled_face_pose'].cuda(args.device) # 6
        real_rh_betas = data['sampled_rh_betas'].cuda(args.device) # 10
        real_rh_pose = data['sampled_rh_pose'].cuda(args.device) # 45
        real_jaw_pose = real_face_pose[:, 3:]

        real_labels = torch.ones((batch_size, 1)).to(args.device)

        # forward-pass
        pred_camera_temp, pred_3d_hand_kps, pred_3d_head_kps, pred_3d_hand_vs_sub, pred_3d_hand_vs, pred_3d_head_vs_sub2, pred_3d_head_vs_sub1, pred_3d_head_vs, pred_hand_contacts_sub, pred_hand_contacts, pred_head_contacts_sub2, pred_head_contacts_sub1, pred_head_contacts, pred_rh_betas, pred_rh_transl, pred_rh_rot, pred_rh_pose, pred_face_shape, pred_face_exp, pred_face_pose, pred_face_rot, pred_face_transl, pred_deformations_sub2, pred_deformations_sub1, pred_deformations, pred_contact_presence = DICE_model(single_image, rh_ref_kps, head_ref_kps, rh_ref_vs_sub, head_ref_vs_sub2, mvm_mask=mvm_mask_, is_train=True)

        flag = False
        if contains_nan(pred_contact_presence):
            print("pred_presence contains nan")
            flag = True
        if contains_nan(pred_deformations_sub2):
            print("pred_deformations_sub2 contains nan")
            flag = True
        if contains_nan(pred_hand_contacts_sub):
            print("pred_hand_contacts_sub contains nan")
            flag = True
        if contains_nan(pred_head_contacts_sub2):
            print("pred_head_contacts_sub2 contains nan")
            flag = True
        
        if flag == True:
            print(data_index, "data index")
            raise ValueError("contains nan")



        pred_jaw_pose = pred_face_pose[:, 3:]
        gt_jaw_pose = gt_face_pose[:, 3:]

        pred_camera_s = 10 * torch.abs(pred_camera_temp[:,0])
        pred_camera_t = pred_camera_temp[:,1:]
        pred_camera = torch.zeros_like(pred_camera_temp)
        pred_camera[:,0] = pred_camera_s
        pred_camera[:,1:] = pred_camera_t


        image_size = torch.tensor([single_image.shape[2], single_image.shape[3]]).repeat(batch_size,1).float().to(args.device)
        image_size = torch.tensor([single_image.shape[2], single_image.shape[3]]).repeat(batch_size,1).float().to(args.device)

        R = torch.diag(torch.tensor([-1, -1, 1])).unsqueeze(0).repeat(batch_size,1,1).to(args.device)
        T = torch.tensor([0, 0, 1]).unsqueeze(0).repeat(batch_size,1).to(args.device)

        cameras = OrthographicCameras(focal_length = pred_camera_s, principal_point = pred_camera_t, device=args.device, R=R, T=T, in_ndc=False, image_size = image_size)

        pred_3d_head_vs_parametric, pred_3d_head_kps_parametric = flame_forwarding(
            flame_model=head_model,
            head_shape_params=pred_face_shape,
            head_expression_params=pred_face_exp,
            head_pose_params=pred_face_pose,
            head_rotation= pred_face_rot,
            head_transl= pred_face_transl,
            head_scale_params=  torch.ones((batch_size,1)).to(args.device),
            return2d=False,
            return_2d_verts=False
        )

        pred_3d_hand_vs_parametric, pred_3d_hand_kps_parametric = mano_forwarding(
              h_model=hand_model,
              betas=pred_rh_betas,
              transl= pred_rh_transl,
              rot= pred_rh_rot,
              pose=pred_rh_pose,
              return_2d=False,
              return_2d_verts=False
          )


        pred_head_center_parametric = pred_3d_head_vs_parametric.mean(dim=1, keepdim=True)
        pred_3d_head_vs_parametric = pred_3d_head_vs_parametric - pred_head_center_parametric
        pred_3d_hand_vs_parametric = pred_3d_hand_vs_parametric - pred_head_center_parametric
        pred_3d_head_kps_parametric = pred_3d_head_kps_parametric - pred_head_center_parametric
        pred_3d_hand_kps_parametric = pred_3d_hand_kps_parametric - pred_head_center_parametric

        pred_head_center_np = pred_3d_head_vs.mean(dim=1, keepdim=True)
        pred_3d_head_vs = pred_3d_head_vs - pred_head_center_np
        pred_3d_head_vs_sub2 = pred_3d_head_vs_sub2 - pred_head_center_np
        pred_3d_head_vs_sub1 = pred_3d_head_vs_sub1 - pred_head_center_np
        pred_3d_hand_vs = pred_3d_hand_vs - pred_head_center_np
        pred_3d_hand_vs_sub = pred_3d_hand_vs_sub - pred_head_center_np
        pred_3d_head_kps = pred_3d_head_kps - pred_head_center_np
        pred_3d_hand_kps = pred_3d_hand_kps - pred_head_center_np


        # get 3d kps from face and hand model
        pred_3d_head_kps_from_model = head_model.convert_vs2landmarks(pred_3d_head_vs.float())
        pred_3d_hand_kps_from_model = mano_model.get_3d_joints(pred_3d_hand_vs.float())

        # obtain 2d joints, which are projected from 3d joints of smpl mesh
        pred_2d_hand_kps = cameras.transform_points_screen(pred_3d_hand_kps)[:, :, :2]
        pred_2d_head_kps = cameras.transform_points_screen(pred_3d_head_kps)[:, :, :2]
        pred_2d_hand_kps_from_model = cameras.transform_points_screen(pred_3d_hand_kps_from_model)[:, :, :2]
        pred_2d_head_kps_from_model = cameras.transform_points_screen(pred_3d_head_kps_from_model)[:, :, :2]
        pred_2d_hand_kps_parametric = cameras.transform_points_screen(pred_3d_hand_kps_parametric)[:, :, :2]
        pred_2d_head_kps_parametric = cameras.transform_points_screen(pred_3d_head_kps_parametric)[:, :, :2]

        pred_2d_hand_vs = cameras.transform_points_screen(pred_3d_hand_vs)[:, :, :2]
        pred_2d_head_vs = cameras.transform_points_screen(pred_3d_head_vs)[:, :, :2]

        pred_2d_hand_vs_sub = cameras.transform_points_screen(pred_3d_hand_vs_sub)[:, :, :2]
        pred_2d_head_vs_sub2 = cameras.transform_points_screen(pred_3d_head_vs_sub2)[:, :, :2]

        pred_3d_hand_vs_parametric_sub = hand_sampler.downsample(pred_3d_hand_vs_parametric.double())
        pred_3d_head_vs_parametric_sub2 = head_sampler.downsample(pred_3d_head_vs_parametric.double())

        pred_2d_hand_vs_parametric_sub = cameras.transform_points_screen(pred_3d_hand_vs_parametric_sub.float())[:, :, :2]
        pred_2d_head_vs_parametric_sub2 = cameras.transform_points_screen(pred_3d_head_vs_parametric_sub2.float())[:, :, :2]

        loss_3d_head_kps = keypoint_3d_loss(criterion_keypoints, pred_3d_head_kps, gt_head_keypoints, has_3d_kp, args.device)
        loss_3d_hand_kps = keypoint_3d_loss(criterion_keypoints, pred_3d_hand_kps, gt_hand_keypoints, has_3d_kp, args.device)

        loss_3d_head_kps_parametric = keypoint_3d_loss(criterion_keypoints, pred_3d_head_kps_parametric, gt_head_keypoints, has_3d_kp, args.device)
        loss_3d_hand_kps_parametric = keypoint_3d_loss(criterion_keypoints, pred_3d_hand_kps_parametric, gt_hand_keypoints, has_3d_kp, args.device)

        head_vloss_sub2 = vertices_loss(criterion_vertices, pred_3d_head_vs_sub2, gt_head_vertices_sub2, has_3d_mesh, args.device)
        head_vloss_sub = vertices_loss(criterion_vertices, pred_3d_head_vs_sub1, gt_head_vertices_sub, has_3d_mesh, args.device)
        head_vloss_full = vertices_loss(criterion_vertices, pred_3d_head_vs, gt_head_vertices_undeformed, has_3d_mesh, args.device)
        hand_vloss_sub = vertices_loss(criterion_vertices, pred_3d_hand_vs_sub, gt_hand_vertices_sub, has_3d_mesh, args.device)
        hand_vloss_full = vertices_loss(criterion_vertices, pred_3d_hand_vs, gt_hand_vertices, has_3d_mesh, args.device)

        head_vloss_sub2_trans = vertices_loss(criterion_vertices, pred_3d_head_vs_sub2 - pred_3d_head_vs_sub2.mean(dim=1).unsqueeze(1), gt_head_vertices_sub2 - gt_head_vertices_sub2.mean(dim=1).unsqueeze(1), has_3d_mesh, args.device)
        head_vloss_sub_trans = vertices_loss(criterion_vertices, pred_3d_head_vs_sub1 - pred_3d_head_vs_sub1.mean(dim=1).unsqueeze(1), gt_head_vertices_sub - gt_head_vertices_sub.mean(dim=1).unsqueeze(1), has_3d_mesh, args.device)
        head_vloss_full_trans = vertices_loss(criterion_vertices, pred_3d_head_vs - pred_3d_head_vs.mean(dim=1).unsqueeze(1), gt_head_vertices_undeformed - gt_head_vertices_undeformed.mean(dim=1).unsqueeze(1), has_3d_mesh, args.device)
        hand_vloss_sub_trans = vertices_loss(criterion_vertices, pred_3d_hand_vs_sub - pred_3d_hand_vs_sub.mean(dim=1).unsqueeze(1), gt_hand_vertices_sub - gt_hand_vertices_sub.mean(dim=1).unsqueeze(1), has_3d_mesh, args.device)
        hand_vloss_full_trans = vertices_loss(criterion_vertices, pred_3d_hand_vs - pred_3d_hand_vs.mean(dim=1).unsqueeze(1), gt_hand_vertices - gt_hand_vertices.mean(dim=1).unsqueeze(1), has_3d_mesh, args.device)


        loss_vertices_head_parametric = \
            vertices_loss(criterion_vertices, pred_3d_head_vs_parametric, gt_head_vertices_undeformed, has_3d_mesh, args.device)


        loss_vertices_hand_parametric = \
            vertices_loss(criterion_vertices, pred_3d_hand_vs_parametric, gt_hand_vertices, has_3d_mesh, args.device)


        loss_vertices_head = args.vloss_head_sub2 * head_vloss_sub2 + \
                            args.vloss_head_sub * head_vloss_sub + \
                            args.vloss_head_full * head_vloss_full + \
                            args.vloss_head_sub2 * head_vloss_sub2_trans + \
                            args.vloss_head_sub * head_vloss_sub_trans + \
                            args.vloss_head_full * head_vloss_full_trans

                        
        loss_vertices_hand = args.vloss_hand_sub * hand_vloss_sub + \
                            args.vloss_hand_full * hand_vloss_full + \
                            args.vloss_hand_sub * hand_vloss_sub_trans + \
                            args.vloss_hand_full * hand_vloss_full_trans

        loss_vertices_np = 3 * loss_vertices_hand + loss_vertices_head 
        loss_vertices_p  = 3 * loss_vertices_hand_parametric + loss_vertices_head_parametric

        loss_contacts_head = args.closs_head_sub2 * contact_loss(criterion_contact, pred_head_contacts_sub2, gt_head_contact_sub2, has_contact, args.device) + \
                            args.closs_head_sub * contact_loss(criterion_contact, pred_head_contacts_sub1, gt_head_contact_sub, has_contact, args.device) + \
                            args.closs_head_full * contact_loss(criterion_contact, pred_head_contacts, gt_head_contact, has_contact, args.device)
        
        loss_contacts_hand = args.closs_hand_sub * contact_loss(criterion_contact, pred_hand_contacts_sub, gt_hand_contact_sub, has_contact, args.device) + \
                            args.closs_hand_full * contact_loss(criterion_contact, pred_hand_contacts, gt_hand_contact, has_contact, args.device)
        
        loss_contacts = loss_contacts_hand + loss_contacts_head
        
        loss_reg_3d_head_kps = keypoint_3d_loss(criterion_keypoints, pred_3d_head_kps_from_model, gt_head_keypoints, has_3d_kp, args.device)
        loss_reg_3d_hand_kps = keypoint_3d_loss(criterion_keypoints, pred_3d_hand_kps_from_model, gt_hand_keypoints, has_3d_kp, args.device)
        # compute 2d joint loss
        loss_2d_joints_hand = keypoint_2d_loss(criterion_2d_keypoints, pred_2d_hand_kps, rh_keys_proj_single, has_2d_kp, single_image.shape[2], args.device)  + \
                         keypoint_2d_loss(criterion_2d_keypoints, pred_2d_hand_kps_from_model, rh_keys_proj_single, has_2d_kp, single_image.shape[2], args.device)
                        
        loss_2d_joints_hand_parametric = keypoint_2d_loss(criterion_2d_keypoints, pred_2d_hand_kps_parametric, rh_keys_proj_single, has_2d_kp, single_image.shape[2], args.device)

        loss_2d_joints_head = keypoint_2d_loss(criterion_2d_keypoints, pred_2d_head_kps, head_keys_proj_single, has_2d_kp, single_image.shape[2], args.device) + \
                            keypoint_2d_loss(criterion_2d_keypoints, pred_2d_head_kps_from_model, head_keys_proj_single, has_2d_kp, single_image.shape[2], args.device)
                            
        loss_2d_joints_head_parametric = keypoint_2d_loss(criterion_2d_keypoints, pred_2d_head_kps_parametric, head_keys_proj_single, has_2d_kp, single_image.shape[2], args.device)

        loss_2d_joints_np = 4 * loss_2d_joints_hand + loss_2d_joints_head
        loss_2d_joints_p = 4 * loss_2d_joints_hand_parametric + loss_2d_joints_head_parametric
        # print("DEBUGGING: hand only!")
        
        # loss_3d_joints = loss_3d_hand_kps + loss_3d_head_kps + loss_reg_3d_hand_kps + loss_reg_3d_head_kps
        loss_3d_joints_np = loss_3d_hand_kps + loss_3d_head_kps + loss_reg_3d_hand_kps + loss_reg_3d_head_kps
        loss_3d_joints_p = loss_3d_hand_kps_parametric + loss_3d_head_kps_parametric

        loss_hand_params = ( 
            params_loss(criterion_params, pred_rh_betas, gt_rh_betas, has_params, args.device) + \
            params_loss(criterion_params, pred_rh_pose, gt_rh_pose, has_params, args.device)
        )

        loss_head_params = (
            params_loss(criterion_params, pred_face_shape, gt_face_shape, has_params, args.device) + \
            params_loss(criterion_params, pred_face_exp, gt_face_exp, has_params, args.device) + \
            params_loss(criterion_params, pred_jaw_pose, gt_jaw_pose, has_params, args.device)
        )

        loss_params = (loss_hand_params + loss_head_params) / 2

        loss_touch = args.touch_loss_weight * criterion_touch(chamfer_distance, pred_3d_head_vs_parametric.float(), pred_3d_hand_vs_parametric.float(), pred_head_contacts, pred_hand_contacts) 
        loss_collision = args.collision_loss_weight * criterion_collision(chamfer_distance, (pred_3d_head_vs_parametric - pred_deformations).float(), pred_3d_hand_vs_parametric.float(), flame_faces)


        if args.normal_vector_loss == "true":
            loss_normal_vector = args.normal_vector_loss_weight * normal_vector_loss_cosine(pred_3d_head_vs_parametric - pred_deformations, gt_head_vertices_deformed, flame_faces)
        else:
            loss_normal_vector = torch.tensor(0).float().cuda(args.device)

        if args.edge_length_loss == "true":
            loss_edge_length = args.edge_length_loss_weight * edge_length_loss(pred_3d_head_vs_parametric - pred_deformations, gt_head_vertices_deformed, flame_faces)
        else:
            loss_edge_length = torch.tensor(0).float().cuda(args.device)

        loss_3d_joints_np = 4 * args.joints_loss_weight_3d * loss_3d_joints_np
        loss_3d_joints_p = args.joints_loss_weight_3d * loss_3d_joints_p
        loss_vertices_np =4 * args.vertices_loss_weight * loss_vertices_np
        loss_vertices_p = args.vertices_loss_weight * loss_vertices_p
        loss_2d_joints_np = args.joints_loss_weight_2d * loss_2d_joints_np
        loss_2d_joints_p = args.joints_loss_weight_2d * loss_2d_joints_p
        loss_contacts = args.contacts_loss_weight * loss_contacts
        loss_params = args.params_loss_weight * loss_params

        if args.self_supervision:
            print("self supervision loss on")
            loss_self_sup_hand = vertices_loss(criterion_vertices, pred_3d_hand_vs_parametric, pred_3d_hand_vs, has_3d_mesh, args.device)
            loss_self_sup_face = vertices_loss(criterion_vertices, pred_3d_head_vs_parametric, pred_3d_head_vs, has_3d_mesh, args.device)
            loss_self_sup = args.vertices_loss_weight * (3 * loss_self_sup_hand + loss_self_sup_face)
        else:
            loss_self_sup = torch.tensor(0).float().cuda(args.device)

        loss_deform, loss_deform_deviation, loss_deform_reg = deform_loss(criterion_deformation, pred_deformations, gt_deformation, args.deform_reg_weight, deform_weight_mask, has_deform, args.device)
        loss_deform_inthewild = deform_reg_loss(criterion_deformation_reg, pred_deformations, gt_deformation, args.deform_reg_weight, deform_weight_mask, has_deform_reg, args.device)

        loss_deform = args.deformation_loss_weight * loss_deform
        loss_deform_deviation = args.deformation_loss_weight * loss_deform_deviation
        loss_deform_reg = args.deformation_loss_weight * loss_deform_reg

        loss_deform_inthewild = args.inthewild_deformation_loss_weight * loss_deform_inthewild.float()

        loss_deform = loss_deform + loss_deform_inthewild

        loss_presence = args.presence_loss_weight * presence_loss(criterion_presence, pred_contact_presence.float(), gt_contact_presence.float(), has_presence, args.device)

        real_labels = torch.ones((batch_size, 1)).to(args.device)

        hand_D_outputs = hand_discriminator(torch.cat([pred_rh_pose, pred_rh_betas], dim=1))
        face_D_outputs = face_discriminator(torch.cat([pred_jaw_pose, pred_face_exp, pred_face_shape], dim=1))

        loss_hand_discriminator = args.discriminator_loss_weight * discriminator_loss(criterion_discriminator, hand_D_outputs, real_labels, has_discr, args.device)
        loss_face_discriminator = args.discriminator_loss_weight * discriminator_loss(criterion_discriminator, face_D_outputs, real_labels, has_discr, args.device)

        loss_discriminator = loss_hand_discriminator + loss_face_discriminator

        ### ------------------- rendering loss starts ------------------- ###

        raster_settings = RasterizationSettings(
                image_size=single_image.shape[2], 
                blur_radius=0.0, 
                faces_per_pixel=1, 
            )
        rasterizer=MeshRasterizer(
                        cameras=cameras, 
                        raster_settings=raster_settings
                    ).to(args.device)
        vertices = merge_verts(pred_3d_hand_vs_parametric, pred_3d_head_vs_parametric).float()
        faces = merge_faces(torch.from_numpy(hand_model.faces.astype(np.int32)), flame_faces).float().to(vertices.device)
        faces = faces.repeat(batch_size, 1, 1)
        mesh = Meshes(verts=vertices, faces=faces)
        fragments = rasterizer(mesh)
        pred_depth_map = fragments.zbuf.squeeze(-1)

        if epoch >= args.burn_in_epochs:
            pred_indices = torch.round(torch.cat([pred_2d_hand_kps_parametric, pred_2d_head_kps_parametric], dim=1)).long()
            gt_indices = torch.round(torch.cat([rh_keys_proj_single, head_keys_proj_single], dim=1)).long()
            loss_depth = args.depth_loss_weight * depth_loss_kp(criterion_depth, pred_depth_map, gt_depth_maps, pred_indices, gt_indices, has_depth, args.device)
        else:
            loss_depth = torch.FloatTensor(1).fill_(0.).to(args.device)

        if args.local_rank == 0:
            with torch.no_grad():
                random_hand_input = torch.randn(batch_size, 45+10).to(args.device)
                random_face_input = torch.randn(batch_size, 100+50+3).to(args.device)
                random_hand_prob = hand_discriminator(random_hand_input).mean()
                random_face_prob = face_discriminator(random_face_input).mean()
                gt_hand_input = torch.cat([gt_rh_pose, gt_rh_betas], dim=1)
                gt_face_input = torch.cat([gt_jaw_pose, gt_face_exp, gt_face_shape], dim=1)
                gt_hand_prob = hand_discriminator(gt_hand_input).mean()
                gt_face_prob = face_discriminator(gt_face_input).mean()

        loss = loss_3d_joints_np + loss_3d_joints_p + loss_vertices_np + loss_vertices_p + loss_2d_joints_np + loss_2d_joints_p + loss_contacts + loss_params + loss_deform + loss_collision + loss_touch + loss_normal_vector + loss_edge_length + loss_presence + loss_discriminator + loss_depth + loss_self_sup
        
        log_loss_2djoints_np.update(loss_2d_joints_np.item(), batch_size)
        log_loss_2djoints_p.update(loss_2d_joints_p.item(), batch_size)
        log_loss_3djoints_np.update(loss_3d_joints_np.item(), batch_size)
        log_loss_3djoints_p.update(loss_3d_joints_p.item(), batch_size)
        log_loss_vertices_np.update(loss_vertices_np.item(), batch_size)
        log_loss_vertices_p.update(loss_vertices_p.item(), batch_size)
        log_loss_self_sup.update(loss_self_sup.item(), batch_size)
        log_losses.update(loss.item(), batch_size)
        log_loss_contacts.update(loss_contacts.item(), batch_size)
        log_loss_params.update(loss_params.item(), batch_size)
        log_loss_deforms.update(loss_deform.item(), batch_size)
        log_loss_collision.update(loss_collision.item(), batch_size)
        log_loss_touch.update(loss_touch.item(), batch_size)
        log_loss_edge_length.update(loss_edge_length.item(), batch_size)
        log_loss_normal_vector.update(loss_normal_vector.item(), batch_size)
        log_loss_deform_deviation.update(loss_deform_deviation.item(), batch_size)
        log_loss_deform_reg.update(loss_deform_reg.item(), batch_size)
        log_loss_contact_presence.update(loss_presence.item(), batch_size)
        
        set_requires_grad([face_discriminator, hand_discriminator], False)
        # back prop
        optimizer.zero_grad()
        loss.backward() 
        nn.utils.clip_grad_value_(DICE_model.parameters(), clip_value=0.3)
        optimizer.step()

        set_requires_grad([face_discriminator, hand_discriminator], True)

        real_labels = torch.ones((batch_size, 1)).to(args.device)
        fake_labels = torch.zeros((batch_size, 1)).to(args.device)
        
        hand_D_optimizer.zero_grad()

        real_outputs = hand_discriminator(torch.cat([real_rh_pose, real_rh_betas], dim=1))
        fake_outputs = hand_discriminator(torch.cat([pred_rh_pose, pred_rh_betas], dim=1).detach()) 
        hand_real_loss = 3 * args.discriminator_loss_weight * discriminator_loss(criterion_discriminator, real_outputs, real_labels, has_discr, args.device)
        hand_fake_loss = 3 * args.discriminator_loss_weight * discriminator_loss(criterion_discriminator, fake_outputs, fake_labels, has_discr, args.device)
        hand_discr_loss = hand_real_loss + hand_fake_loss
        if hand_discr_loss.item() > 0:
            hand_discr_loss.backward()
            hand_D_optimizer.step()

        face_real_labels = torch.ones((batch_size, 1)).to(args.device)
        face_fake_labels = torch.zeros((batch_size, 1)).to(args.device)

        face_D_optimizer.zero_grad()

        face_real_outputs = face_discriminator(torch.cat([real_jaw_pose, real_face_exp, real_face_shape], dim=1))
        face_fake_outputs = face_discriminator(torch.cat([pred_jaw_pose, pred_face_exp, pred_face_shape], dim=1).detach()) # need to call detach!
        face_real_loss = args.discriminator_loss_weight * discriminator_loss(criterion_discriminator, face_real_outputs, face_real_labels, has_discr, args.device)
        face_fake_loss = args.discriminator_loss_weight * discriminator_loss(criterion_discriminator, face_fake_outputs, face_fake_labels, has_discr, args.device)
        face_discr_loss = face_real_loss + face_fake_loss
        if face_discr_loss.item() > 0:
            face_discr_loss.backward()
            face_D_optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (iteration == 1 and args.debug_val == "true") or iteration % args.logging_steps == 0 or iteration == max_iter:
            if not os.path.exists(args.output_dir + f"/mesh_{epoch}_{iteration}"):
                try:
                    os.mkdir(args.output_dir + f"/mesh_{epoch}_{iteration}") 
                except FileExistsError:
                    print(args.output_dir + f"/mesh_{epoch}_{iteration} + " + "already exists")
            eta_seconds = batch_time.avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(f"head_vloss_sub2: {head_vloss_sub2}, head_vloss_sub: {head_vloss_sub}, head_vloss_full: {head_vloss_full}, hand_vloss_sub: {hand_vloss_sub}, hand_vloss_full: {hand_vloss_full}")
            log_string = f"eta: {eta_string}, epoch: {epoch}, iter: {iteration}, max mem : {torch.cuda.max_memory_allocated() / 1024.0 / 1024.0:.0f}  loss: {log_losses.avg:.4f}, 2d joint loss nonparametric: {log_loss_2djoints_np.avg:.4f}, 2d joint loss parametric: {log_loss_2djoints_p.avg:.4f}, 3d joint loss nonparametric: {log_loss_3djoints_np.avg:.4f}, 3d joint loss parametric: {log_loss_3djoints_p.avg:.4f}, vertex loss nonparametric: {log_loss_vertices_np.avg:.4f}, vertex loss parametric: {log_loss_vertices_p.avg:.4f}, self supervision loss: {log_loss_self_sup.avg:.4f}, contact loss: {log_loss_contacts.avg:.4f}, edge length loss: {log_loss_edge_length.avg:.4f}, normal vector loss: {log_loss_normal_vector.avg:.4f}, params loss: {log_loss_params.avg:.4f}, deform loss: {log_loss_deforms.avg:.4f}, touch loss: {log_loss_touch.avg:.4f}, collision loss: {log_loss_collision.avg:.4f}, deform deviation loss: {log_loss_deform_deviation.avg:.4f}, deform reg loss: {log_loss_deform_reg.avg:.4f}, contact presence loss: {log_loss_contact_presence.avg:.4f}, compute: {batch_time.avg:.4f}, data: {data_time.avg:.4f}, lr: {optimizer.param_groups[0]['lr']:.6f}"
            logger.info(log_string)
            
            raster_settings = RasterizationSettings(
                image_size=single_image.shape[2], 
                blur_radius=0.0, 
                faces_per_pixel=1, 
            )
            if args.local_rank == 0:
                visual_imgs = []
                gt_kp_imgs = []
                pred_kp_imgs = []
                pred_kp_parametric_imgs = []
                gt_vs_imgs = []
                pred_vs_imgs = []
                pred_vs_parametric_imgs = []
                rasterizer=MeshRasterizer(
                        cameras=cameras, 
                        raster_settings=raster_settings
                    ).to(args.device)
                lights = PointLights(device=args.device, location=[[0.0, 0.0, -3.0]])
                shader=SoftPhongShader(
                    device=args.device, 
                    cameras=cameras,
                    lights=lights
                )
                vertices = merge_verts(pred_3d_hand_vs_parametric, pred_3d_head_vs_parametric - pred_deformations).float()
                faces = merge_faces(torch.from_numpy(hand_model.faces.astype(np.int32)), flame_faces).float().to(vertices.device)
                faces = faces.repeat(batch_size, 1, 1)
                texture_map = torch.ones((batch_size, vertices.shape[1], 3), dtype=torch.float32).to(args.device)  # (N, V, C)
                contact_map = torch.cat([pred_hand_contacts, pred_head_contacts], dim=1).squeeze().detach()
                contact_map = (contact_map - contact_map.max()) / (contact_map.min() - contact_map.max())
                texture_map[:, :, 1] = contact_map
                # Create a TexturesVertex object
                textures = TexturesVertex(verts_features=texture_map)
                mesh = Meshes(verts=vertices, faces=faces, textures=textures)
                fragments = rasterizer(mesh)
                images = shader(meshes=mesh, fragments=fragments)
                zbuf = fragments.zbuf.squeeze() # B, 224, 224
                image_p = images[..., :3].detach().cpu().numpy()
                # print(zbuf.shape, depth_map.shape, image.shape)
                image_p = image_p[:min(batch_size, 10), :, :, :]*255
                image_p = image_p.reshape(-1, single_image.shape[2], 3)
                
                # render non-parametric mesh
                vertices = merge_verts(pred_3d_hand_vs, pred_3d_head_vs - pred_deformations).float()
                textures = TexturesVertex(verts_features=texture_map)
                mesh = Meshes(verts=vertices, faces=faces, textures=textures)
                fragments = rasterizer(mesh)
                images = shader(meshes=mesh, fragments=fragments)
                image_np = images[..., :3].detach().cpu().numpy()
                image_np = image_np[:min(batch_size, 10), :, :, :]*255
                image_np = image_np.reshape(-1, single_image.shape[2], 3)
                print("rendering complete")
                for i in range(min(batch_size, 10)):
                    img = visualize_keypoints_single(single_image[i], rh_keys_proj_single[i], head_keys_proj_single[i])
                    gt_kp_imgs.append(img)
                    img = visualize_keypoints_single(single_image[i], pred_2d_hand_kps_from_model[i].detach() / single_image.shape[2], pred_2d_head_kps_from_model[i].detach() / single_image.shape[2], color1=(0,255,0), color2=(0,0,255))
                    pred_kp_imgs.append(img)
                    img = visualize_keypoints_single(single_image[i], pred_2d_hand_kps_parametric[i].detach() / single_image.shape[2], pred_2d_head_kps_parametric[i].detach() / single_image.shape[2], color1=(0,255,0), color2=(0,0,255))
                    pred_kp_parametric_imgs.append(img)
                    img = visualize_keypoints_single(single_image[i], rh_vs_proj_single_sub[i], head_vs_proj_single_sub2[i])
                    gt_vs_imgs.append(img)
                    img = visualize_keypoints_single(single_image[i], pred_2d_hand_vs_sub[i].detach() / single_image.shape[2], pred_2d_head_vs_sub2[i].detach() / single_image.shape[2], color1=(0,255,0), color2=(0,0,255))
                    pred_vs_imgs.append(img)
                    img = visualize_keypoints_single(single_image[i], pred_2d_hand_vs_parametric_sub[i].detach() / single_image.shape[2], pred_2d_head_vs_parametric_sub2[i].detach() / single_image.shape[2], color1=(0,255,0), color2=(0,0,255))
                    pred_vs_parametric_imgs.append(img)

                    save_mesh(args.output_dir + f"/mesh_{epoch}_{iteration}/" + f"{i}_pred_head_mesh_{data_index[i]}.obj", pred_3d_head_vs[i].cpu().detach().numpy(), flame_faces)
                    save_mesh(args.output_dir + f"/mesh_{epoch}_{iteration}/" + f"{i}_pred_hand_mesh_{data_index[i]}.obj", pred_3d_hand_vs[i].cpu().detach().numpy(), hand_model.faces)
                    save_mesh(args.output_dir + f"/mesh_{epoch}_{iteration}/" + f"{i}_pred_head_mesh_parametric_undeformed_{data_index[i]}.obj", pred_3d_head_vs_parametric[i].cpu().detach().numpy(), flame_faces)
                    save_mesh(args.output_dir + f"/mesh_{epoch}_{iteration}/" + f"{i}_pred_hand_mesh_parametric_{data_index[i]}.obj", pred_3d_hand_vs_parametric[i].cpu().detach().numpy(), hand_model.faces)
                    save_mesh(args.output_dir + f"/mesh_{epoch}_{iteration}/" + f"{i}_pred_head_mesh_parametric_deformed_{data_index[i]}.obj", (pred_3d_head_vs_parametric[i] - pred_deformations[i]).cpu().detach().numpy(), flame_faces)
                    save_mesh(args.output_dir + f"/mesh_{epoch}_{iteration}/" + f"{i}_gt_head_mesh_{data_index[i]}.obj", gt_head_vertices_deformed[i].cpu().detach().numpy(), flame_faces)
                    save_mesh(args.output_dir + f"/mesh_{epoch}_{iteration}/" + f"{i}_gt_hand_mesh_{data_index[i]}.obj", gt_hand_vertices[i].cpu().detach().numpy(), hand_model.faces)

                    save_mesh(args.output_dir + f"/mesh_{epoch}_{iteration}/" + f"{i}_head_kp_cam_{data_index[i]}.obj", gt_head_keypoints[i].cpu().detach().numpy(), np.array([]))
                    save_mesh(args.output_dir + f"/mesh_{epoch}_{iteration}/" + f"{i}_hand_kp_cam_{data_index[i]}.obj", gt_hand_keypoints[i].cpu().detach().numpy(), np.array([]))
                    save_mesh(args.output_dir + f"/mesh_{epoch}_{iteration}/" + f"{i}_head_kp_pred_np_{data_index[i]}.obj", pred_3d_head_kps[i].cpu().detach().numpy(), np.array([]))
                    save_mesh(args.output_dir + f"/mesh_{epoch}_{iteration}/" + f"{i}_head_kp_pred_from_model_{data_index[i]}.obj", pred_3d_head_kps_from_model[i].cpu().detach().numpy(), np.array([]))
                    save_mesh(args.output_dir + f"/mesh_{epoch}_{iteration}/" + f"{i}_head_kp_pred_parametric_{data_index[i]}.obj", pred_3d_head_kps_parametric[i].cpu().detach().numpy(), np.array([]))
                    save_mesh(args.output_dir + f"/mesh_{epoch}_{iteration}/" + f"{i}_hand_kp_pred_np_{data_index[i]}.obj", pred_3d_hand_kps[i].cpu().detach().numpy(), np.array([]))
                    save_mesh(args.output_dir + f"/mesh_{epoch}_{iteration}/" + f"{i}_hand_kp_pred_from_model_{data_index[i]}.obj", pred_3d_hand_kps_from_model[i].cpu().detach().numpy(), np.array([]))
                    save_mesh(args.output_dir + f"/mesh_{epoch}_{iteration}/" + f"{i}_hand_kp_pred_parametric_{data_index[i]}.obj", pred_3d_hand_kps_parametric[i].cpu().detach().numpy(), np.array([]))


                gt_kp_imgs = np.vstack(gt_kp_imgs)
                pred_kp_imgs = np.vstack(pred_kp_imgs)
                gt_vs_imgs = np.vstack(gt_vs_imgs)
                pred_vs_imgs = np.vstack(pred_vs_imgs)
                pred_kp_parametric_imgs = np.vstack(pred_kp_parametric_imgs)
                pred_vs_parametric_imgs = np.vstack(pred_vs_parametric_imgs)
                gt_depth_maps = gt_depth_maps[:min(batch_size, 10), :, :]
                pred_depth_maps = zbuf[:min(batch_size, 10), :, :]
                gt_depth_maps = gt_depth_maps.detach().cpu().numpy() # (N, H, W)
                pred_depth_maps = process_zbuf(pred_depth_maps).detach().cpu().numpy() # (N, H, W)

                gt_depth_maps_vis = depth_to_heatmap_batch(gt_depth_maps)
                pred_depth_maps_vis = depth_to_heatmap_batch(pred_depth_maps)
                gt_depth_maps_vis = np.vstack(gt_depth_maps_vis)
                pred_depth_maps_vis = np.vstack(pred_depth_maps_vis)
                gt_depth_maps_vis = (gt_depth_maps_vis * 255).astype(np.uint8)
                pred_depth_maps_vis = (pred_depth_maps_vis * 255).astype(np.uint8)

                visual_imgs = np.hstack([gt_kp_imgs, pred_kp_imgs, pred_kp_parametric_imgs, gt_vs_imgs, pred_vs_imgs, pred_vs_parametric_imgs, image_p, image_np, gt_depth_maps_vis, pred_depth_maps_vis])


            if args.local_rank == 0:
                stamp = str(epoch) + '_' + str(iteration)
                temp_fname = args.output_dir + 'visual_' + stamp + '.jpg'
                cv2.imwrite(temp_fname, visual_imgs)

        # print("visualization finished")

        if val_dataloader is None and (iteration-1) % 1000 == 0:
            checkpoint_dir = save_checkpoint(DICE_model, args, epoch, iteration, face_d_model=face_discriminator, hand_d_model=hand_discriminator)

        if val_dataloader is not None and ( (iteration == 1 and args.debug_val == "true") or iteration % iters_per_epoch == 0 ): 
            val_mPVE, val_mPVE_trans, val_mPJPE, val_PAmPJPE, val_count, val_Touchness, val_NonColRatio, val_ColDist, Contact_precision, Contact_recall, Contact_accuracy, Contact_f1_score, hand_precision, hand_recall, hand_accuracy, hand_f1_score, head_precision, head_recall, head_accuracy, head_f1_score, presence_precision, presence_recall, presence_accuracy, presence_f1_score, val_DeformError, val_LargeDeformError \
            = run_validate(args, val_dataloader,
                                                DICE_model, 
                                                criterion_keypoints, 
                                                criterion_vertices, 
                                                epoch, 
                                                mano_model,
                                                hand_model,
                                                head_model,
                                                hand_sampler=hand_sampler,
                                                head_sampler=head_sampler,
                                                train_iteration=iteration)

            val_FScore = 2*val_Touchness*val_NonColRatio/(val_Touchness+val_NonColRatio)

            logger.info(
                f'Validation epoch: {epoch}  mPVE: {1000*val_mPVE:6.2f}, mPVE_trans: {1000*val_mPVE_trans:6.2f}, mPJPE: {1000*val_mPJPE:6.2f}, PAmPJPE: {1000*val_PAmPJPE:6.2f}, ColDist: {1000*val_ColDist:6.2f}, Touchness: {100*val_Touchness:6.2f}, NonColRatio: {100*val_NonColRatio:6.2f}, F-Score: {100*val_FScore:6.2f}, Contact_precision: {Contact_precision:6.2f}, Contact_recall: {Contact_recall:6.2f}, Contact_accuracy: {Contact_accuracy:6.2f}, Contact_f1_score: {Contact_f1_score:6.2f}, hand_precision: {hand_precision:6.2f}, hand_recall: {hand_recall:6.2f}, hand_accuracy: {hand_accuracy:6.2f}, hand_f1_score: {hand_f1_score:6.2f}, head_precision: {head_precision:6.2f}, head_recall: {head_recall:6.2f}, head_accuracy: {head_accuracy:6.2f}, head_f1_score: {head_f1_score:6.2f}, presence_precision: {presence_precision:6.2f}, presence_recall: {presence_recall:6.2f}, presence_accuracy: {presence_accuracy:6.2f}, presence_f1_score: {presence_f1_score:6.2f}, deform error:{1000*val_DeformError:6.2f}, large deform error:{1000*val_LargeDeformError:6.2f}, Data Count: {val_count:6.2f}'
            )


            if val_mPVE < log_eval_metrics.mPVE and val_mPVE_trans < log_eval_metrics.mPVE_trans:
                log_eval_metrics.update(mPVE=val_mPVE, mPVE_trans=val_mPVE_trans, mPJPE=val_mPJPE,
                                        PAmPJPE=val_PAmPJPE, ColDist=val_ColDist, Touchness=val_Touchness, NonColRatio=val_NonColRatio, FScore=val_FScore, epoch=epoch)
                # val_mPJPE, val_PAmPJPE, val_ColDist, val_Touchness, val_NonColRatio, val_FScore, epoch)
            checkpoint_dir = save_checkpoint(DICE_model, args, epoch, iteration, face_d_model=face_discriminator, hand_d_model=hand_discriminator)
                
        
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info('Total training time: {} ({:.4f} s / iter)'.format(
        total_time_str, total_training_time / max_iter)
    )
    checkpoint_dir = save_checkpoint(DICE_model, args, epoch, iteration, face_d_model=face_discriminator, hand_d_model=hand_discriminator)

    logger.info(
        ' Best Results:'
        + '  mPVE: {:6.2f}, mPJPE: {:6.2f}, PAmPJPE: {:6.2f}, at epoch {:6.2f}'.format(1000*log_eval_metrics.mPVE, 1000*log_eval_metrics.mPJPE, 1000*log_eval_metrics.PAmPJPE, log_eval_metrics.epoch)
    )


def run_eval_general(args, val_dataloader, DICE_model, smpl, mesh_sampler):
    smpl.eval()
    criterion_keypoints = torch.nn.MSELoss(reduction='none').cuda(args.device)
    criterion_vertices = torch.nn.L1Loss().cuda(args.device)

    epoch = 0
    if args.distributed:
        DICE_model = torch.nn.parallel.DistributedDataParallel(
            DICE_model, device_ids=[args.local_rank], 
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    DICE_model.eval()

    val_mPVE, val_mPJPE, val_PAmPJPE, val_count = run_validate(args, val_dataloader, 
                                    DICE_model, 
                                    criterion_keypoints, 
                                    criterion_vertices, 
                                    epoch, 
                                    smpl,
                                    mesh_sampler)

    logger.info(
        ' '.join(['Validation', 'epoch: {ep}',]).format(ep=epoch) 
        + '  mPVE: {:6.2f}, mPJPE: {:6.2f}, PAmPJPE: {:6.2f} '.format(1000*val_mPVE, 1000*val_mPJPE, 1000*val_PAmPJPE)
    )
    return

def calculate_metrics(predicted_probs, true_labels, threshold=0.5):
    # Convert predicted probabilities to binary predictions
    predicted_binary = (predicted_probs >= threshold).float()

    # Calculate TP, TN, FP, FN
    TP = torch.sum((predicted_binary == 1) & (true_labels == 1)).item()
    TN = torch.sum((predicted_binary == 0) & (true_labels == 0)).item()
    FP = torch.sum((predicted_binary == 1) & (true_labels == 0)).item()
    FN = torch.sum((predicted_binary == 0) & (true_labels == 1)).item()

    return TP, TN, FP, FN

def run_validate(args, val_loader, DICE_model, criterion, criterion_vertices, epoch, mano_model, hand_model, head_model, head_sampler=None, hand_sampler=None, train_iteration=None):
    batch_time = AverageMeter()
    mPVE = AverageMeter()
    mPVE_trans = AverageMeter()
    mPJPE = AverageMeter()
    PAmPJPE = AverageMeter()
    ColDist = AverageMeter()
    NonColRatio = AverageMeter()
    Touchness = AverageMeter()
    DeformError = AverageMeter()
    LargeDeformError = AverageMeter()
    # switch to evaluate mode
    DICE_model.eval()
    mano_model.eval()
    hand_model.eval()
    head_model.eval()

    asset_path = args.data_path+"/assets/"
    flame_model_path = asset_path+"/generic_model.pkl"
    flame_faces = get_FLAME_faces(flame_model_path)

    global_TP, global_TN, global_FP, global_FN = 0, 0, 0, 0
    global_hand_TP, global_hand_TN, global_hand_FP, global_hand_FN = 0, 0, 0, 0
    global_face_TP, global_face_TN, global_face_FP, global_face_FN = 0, 0, 0, 0
    global_presence_TP, global_presence_TN, global_presence_FP, global_presence_FN = 0, 0, 0, 0

    avg = []
    gt_kp_imgs = []
    pred_kp_imgs = []
    pred_kp_parametric_imgs = []
    gt_vs_imgs = []
    pred_vs_imgs = []
    pred_vs_parametric_imgs = []
    render_imgs = []

    neck_idx = np.load(args.data_path+"/assets/neck_idx.npy")

    try:
        os.mkdir(args.output_dir + f"/val_mesh_epoch{epoch}") 
    except FileExistsError:
        print(args.output_dir + f"/val_mesh_epoch{epoch} " + "already exists")

    with torch.no_grad():
        # end = time.time()
        print("validating...")
        for iteration, data in tqdm(enumerate(val_loader)):
            mode = data["mode"]
            sub = data["sub_id"]
            cam = data["cam_id"]
            frame = data["frame_id"] 

            batch_size = data["single_img_seqs"].shape[0]

            rh_ref_vs = data["rh_ref_vs"].cuda(args.device)
            head_ref_vs = data["head_ref_vs"].cuda(args.device)

            head_ref_vs_sub2 = head_sampler.downsample(head_ref_vs.double()).float()
            rh_ref_vs_sub = hand_sampler.downsample(rh_ref_vs.double()).float()
            head_ref_kps = head_model.convert_vs2landmarks(head_ref_vs)
            rh_ref_kps = mano_model.get_3d_joints(rh_ref_vs)

            single_image = data["single_img_seqs"].cuda(args.device)


            gt_hand_contact = data["rh_con_labels"].cuda(args.device).unsqueeze(-1)
            gt_head_contact = data["head_con_labels"].cuda(args.device).unsqueeze(-1)

            gt_hand_vertices = data["rh_vs_in_cam"].cuda(args.device)
            gt_head_vertices_undeformed = data["head_vs_in_cam"].cuda(args.device)
            gt_hand_keypoints = data["rh_keys_in_cam"].cuda(args.device)
            gt_head_keypoints = data["head_keys_in_cam"].cuda(args.device)
            gt_head_vertices_deformed = data["head_vs_in_cam_deformed"].cuda(args.device)
            gt_deformation = data["deformation_cam"].cuda(args.device)

            gt_head_center = gt_head_vertices_deformed.mean(dim=1)

            gt_head_vertices_undeformed = gt_head_vertices_undeformed - gt_head_center[:, None, :]
            gt_head_vertices_deformed = gt_head_vertices_deformed - gt_head_center[:, None, :]
            gt_hand_vertices = gt_hand_vertices - gt_head_center[:, None, :]
            gt_hand_keypoints = gt_hand_keypoints - gt_head_center[:, None, :]
            gt_head_keypoints = gt_head_keypoints - gt_head_center[:, None, :]
            
            head_vs_proj_single = data["head_vs_proj_single"].cuda(args.device)
            rh_vs_proj_single = data["rh_vs_proj_single"].cuda(args.device)
            head_keys_proj_single = data["head_keys_proj_single"].cuda(args.device)
            rh_keys_proj_single = data["rh_keys_proj_single"].cuda(args.device)

            head_vs_proj_single_sub2 = head_sampler.downsample(head_vs_proj_single.double())
            rh_vs_proj_single_sub = hand_sampler.downsample(rh_vs_proj_single.double())

            gt_contact_presence, _ = torch.max(torch.cat([gt_hand_contact, gt_head_contact], dim=1), dim=1)

            pred_camera_temp, pred_3d_hand_kps, pred_3d_head_kps, pred_3d_hand_vs_sub, pred_3d_hand_vs, pred_3d_head_vs_sub2, pred_3d_head_vs_sub1, pred_3d_head_vs, pred_hand_contacts_sub, pred_hand_contacts, pred_head_contacts_sub2, pred_head_contacts_sub1, pred_head_contacts, pred_rh_betas, pred_rh_transl, pred_rh_rot, pred_rh_pose, pred_face_shape, pred_face_exp, pred_face_pose, pred_face_rot, pred_face_transl, pred_deformations_sub2, pred_deformations_sub1, pred_deformations, pred_contact_presence = DICE_model(single_image, rh_ref_kps, head_ref_kps, rh_ref_vs_sub, head_ref_vs_sub2, is_train=False)

            pred_camera_s = 10 * torch.abs(pred_camera_temp[:,0])
            pred_camera_t = pred_camera_temp[:,1:]
            pred_camera = torch.zeros_like(pred_camera_temp)
            pred_camera[:,0] = pred_camera_s
            pred_camera[:,1:] = pred_camera_t

            image_size = torch.tensor([single_image.shape[2], single_image.shape[3]]).repeat(batch_size,1).float().to(args.device)

            R = torch.diag(torch.tensor([-1, -1, 1])).unsqueeze(0).repeat(batch_size,1,1).to(args.device)
            T = torch.tensor([0, 0, 1]).unsqueeze(0).repeat(batch_size,1).to(args.device)

            cameras = OrthographicCameras(focal_length = pred_camera_s, principal_point = pred_camera_t, device=args.device, R=R, T=T, in_ndc=False, image_size = image_size)

            pred_3d_head_vs_parametric, pred_3d_head_kps_parametric = flame_forwarding(
                flame_model=head_model,
                head_shape_params=pred_face_shape,
                head_expression_params=pred_face_exp,
                head_pose_params=pred_face_pose,
                head_rotation= pred_face_rot,
                head_transl= pred_face_transl,
                head_scale_params=  torch.ones((batch_size,1)).to(pred_face_shape.device),
                return2d=False,
                return_2d_verts=False
            )

            pred_3d_hand_vs_parametric, pred_3d_hand_kps_parametric = mano_forwarding(
                h_model=hand_model,
                betas=pred_rh_betas,
                transl= pred_rh_transl,
                rot= pred_rh_rot,
                pose=pred_rh_pose,
                return_2d=False,
                return_2d_verts=False
            )

            pred_head_center_parametric = pred_3d_head_vs_parametric.mean(dim=1, keepdim=True)
            pred_3d_head_vs_parametric = pred_3d_head_vs_parametric - pred_head_center_parametric
            pred_3d_hand_vs_parametric = pred_3d_hand_vs_parametric - pred_head_center_parametric
            pred_3d_head_kps_parametric = pred_3d_head_kps_parametric - pred_head_center_parametric
            pred_3d_hand_kps_parametric = pred_3d_hand_kps_parametric - pred_head_center_parametric

            pred_head_center_np = pred_3d_head_vs.mean(dim=1, keepdim=True)
            pred_3d_head_vs = pred_3d_head_vs - pred_head_center_np
            pred_3d_head_vs_sub2 = pred_3d_head_vs_sub2 - pred_head_center_np
            pred_3d_head_vs_sub1 = pred_3d_head_vs_sub1 - pred_head_center_np
            pred_3d_hand_vs = pred_3d_hand_vs - pred_head_center_np
            pred_3d_hand_vs_sub = pred_3d_hand_vs_sub - pred_head_center_np
            pred_3d_head_kps = pred_3d_head_kps - pred_head_center_np
            pred_3d_hand_kps = pred_3d_hand_kps - pred_head_center_np


            pred_3d_head_vs_parametric = pred_3d_head_vs_parametric - pred_deformations

            pred_3d_head_kps_from_model = head_model.convert_vs2landmarks(pred_3d_head_vs_parametric.float()) # get keypoints from deformed mesh
            pred_3d_head_kps_from_model = head_model.convert_vs2landmarks(pred_3d_head_vs.float())
            pred_3d_hand_kps_from_model = mano_model.get_3d_joints(pred_3d_hand_vs.float())
            
            pred_2d_hand_kps = cameras.transform_points_screen(pred_3d_hand_kps)[:, :, :2] / single_image.shape[2]
            pred_2d_head_kps = cameras.transform_points_screen(pred_3d_head_kps)[:, :, :2] / single_image.shape[2]
            pred_2d_hand_kps_from_model = cameras.transform_points_screen(pred_3d_hand_kps_from_model)[:, :, :2] / single_image.shape[2]
            pred_2d_head_kps_from_model = cameras.transform_points_screen(pred_3d_head_kps_from_model)[:, :, :2] / single_image.shape[2]
            pred_2d_hand_kps_parametric = cameras.transform_points_screen(pred_3d_hand_kps_parametric)[:, :, :2] / single_image.shape[2]
            pred_2d_head_kps_parametric = cameras.transform_points_screen(pred_3d_head_kps_parametric)[:, :, :2] / single_image.shape[2]

            pred_2d_hand_vs = cameras.transform_points_screen(pred_3d_hand_vs)[:, :, :2] / single_image.shape[2]
            pred_2d_head_vs = cameras.transform_points_screen(pred_3d_head_vs)[:, :, :2] / single_image.shape[2]

            pred_2d_hand_vs_sub = cameras.transform_points_screen(pred_3d_hand_vs_sub)[:, :, :2] / single_image.shape[2]
            pred_2d_head_vs_sub2 = cameras.transform_points_screen(pred_3d_head_vs_sub2)[:, :, :2] / single_image.shape[2]

            pred_3d_hand_vs_parametric_sub = hand_sampler.downsample(pred_3d_hand_vs_parametric.double())
            pred_3d_head_vs_parametric_sub2 = head_sampler.downsample(pred_3d_head_vs_parametric.double())

            pred_2d_hand_vs_parametric_sub = cameras.transform_points_screen(pred_3d_hand_vs_parametric_sub.float())[:, :, :2] / single_image.shape[2]
            pred_2d_head_vs_parametric_sub2 = cameras.transform_points_screen(pred_3d_head_vs_parametric_sub2.float())[:, :, :2] / single_image.shape[2]

            pred_deformations[:, neck_idx, :] = 0.0
            print("clean deformations")

            if iteration % 5 == 0:
                i = 0
                raster_settings = RasterizationSettings(
                    image_size=single_image.shape[2], 
                    blur_radius=0.0, 
                    faces_per_pixel=1, 
                )
                rasterizer=MeshRasterizer(
                        cameras=cameras[0], 
                        raster_settings=raster_settings
                    ).to(args.device)
                lights = PointLights(device=args.device, location=[[0.0, 0.0, -3.0]])
                shader=SoftPhongShader(
                    device=args.device, 
                    cameras=cameras[0],
                    lights=lights
                )
                vertices = merge_verts(pred_3d_hand_vs_parametric[i].unsqueeze(0), (pred_3d_head_vs_parametric[i] - pred_deformations[i]).unsqueeze(0)).float()
                faces = merge_faces(torch.from_numpy(hand_model.faces.astype(np.int32)), flame_faces).float().to(vertices.device)
                faces = faces.unsqueeze(0)
                texture_map = torch.ones((1, vertices.shape[1], 3), dtype=torch.float32).to(args.device)  # (N, V, C)
                textures = TexturesVertex(verts_features=texture_map)
                mesh = Meshes(verts=vertices, faces=faces, textures=textures)
                fragments = rasterizer(mesh)
                images = shader(meshes=mesh, fragments=fragments)
                image = images[..., :3].detach().cpu().numpy()
                image = image[0, :, :, :]*255
                image = image.reshape(-1, single_image.shape[2], 3)
                render_imgs.append(image)
                img = visualize_keypoints_single(single_image[i], rh_keys_proj_single[i], head_keys_proj_single[i])
                img = overlay_number(img, f"{iteration} {sub[i]} {cam[i]} {frame[i]}")
                gt_kp_imgs.append(img)
                img = visualize_keypoints_single(single_image[i], pred_2d_hand_kps_from_model[i].detach(), pred_2d_head_kps_from_model[i].detach(), color1=(0,255,0), color2=(0,0,255))
                pred_kp_imgs.append(img)
                img = visualize_keypoints_single(single_image[i], pred_2d_hand_kps_parametric[i].detach(), pred_2d_head_kps_parametric[i].detach(), color1=(0,255,0), color2=(0,0,255))
                pred_kp_parametric_imgs.append(img)
                img = visualize_keypoints_single(single_image[i], rh_vs_proj_single_sub[i], head_vs_proj_single_sub2[i])
                gt_vs_imgs.append(img)
                img = visualize_keypoints_single(single_image[i], pred_2d_hand_vs_sub[i].detach(), pred_2d_head_vs_sub2[i].detach(), color1=(0,255,0), color2=(0,0,255))
                pred_vs_imgs.append(img)
                img = visualize_keypoints_single(single_image[i], pred_2d_hand_vs_parametric_sub[i].detach(), pred_2d_head_vs_parametric_sub2[i].detach(), color1=(0,255,0), color2=(0,0,255))
                pred_vs_parametric_imgs.append(img)
                save_mesh(args.output_dir + f"/val_mesh_epoch{epoch}/" + f"{iteration}_pred_head_mesh_parametric_undeformed.obj", (pred_3d_head_vs_parametric[i] + pred_deformations[i]).cpu().detach().numpy(), flame_faces)
                save_mesh(args.output_dir + f"/val_mesh_epoch{epoch}/" + f"{iteration}_pred_head_mesh_parametric_deformed.obj", pred_3d_head_vs_parametric[i].cpu().detach().numpy(), flame_faces)
                save_mesh(args.output_dir + f"/val_mesh_epoch{epoch}/" + f"{iteration}_pred_hand_mesh_parametric.obj", pred_3d_hand_vs_parametric[i].cpu().detach().numpy(), hand_model.faces)
                save_mesh(args.output_dir + f"/val_mesh_epoch{epoch}/" + f"{iteration}_pred_head_mesh.obj", pred_3d_head_vs[i].cpu().detach().numpy(), flame_faces)
                save_mesh(args.output_dir + f"/val_mesh_epoch{epoch}/" + f"{iteration}_pred_head_mesh_deformed.obj", (pred_3d_head_vs[i] - pred_deformations[i]).cpu().detach().numpy(), flame_faces)
                save_mesh(args.output_dir + f"/val_mesh_epoch{epoch}/" + f"{iteration}_pred_hand_mesh.obj", pred_3d_hand_vs[i].cpu().detach().numpy(), hand_model.faces)
                save_mesh(args.output_dir + f"/val_mesh_epoch{epoch}/" + f"{iteration}_gt_head_mesh.obj", gt_head_vertices_deformed[i].cpu().detach().numpy(), flame_faces)
                save_mesh(args.output_dir + f"/val_mesh_epoch{epoch}/" + f"{iteration}_gt_hand_mesh.obj", gt_hand_vertices[i].cpu().detach().numpy(), hand_model.faces)

            batch_non_col_count, batch_non_touching_count, batch_gt_touch_count, batch_collision_distance_sum, batch_count = get_plausibility_metrics(gt_head_vertices_deformed.cpu().numpy(), gt_hand_vertices.cpu().numpy(), pred_3d_head_vs_parametric.cpu().numpy(), pred_3d_hand_vs_parametric.cpu().numpy(), flame_faces) # using parametric mesh for plausibility metrics

            pred_vertices = torch.cat([pred_3d_head_vs_parametric, pred_3d_hand_vs_parametric], dim=1)
            gt_vertices = torch.cat([gt_head_vertices_deformed, gt_hand_vertices], dim=1)

            pred_joints = torch.cat([pred_3d_head_kps_parametric, pred_3d_hand_kps_parametric], dim=1) # using joints regressed from deformed mesh for joint metrics
            gt_joints = torch.cat([gt_head_keypoints, gt_hand_keypoints], dim=1) # already subtracted head center

            # measure errors
            error_deform, error_deform_large = deform_error(pred_deformations, gt_deformation)
            error_vertices = mean_per_vertex_error(pred_vertices, gt_vertices)
            error_vertices_trans = mean_per_vertex_error(pred_vertices, gt_vertices)
            error_joints = mean_per_joint_position_error(pred_joints, gt_joints)
            error_joints_pa = reconstruction_error(pred_joints.cpu().numpy(), gt_joints.cpu().numpy(), reduction=None)

            batch_TP, batch_TN, batch_FP, batch_FN = 0, 0, 0, 0
            batch_face_TP, batch_face_TN, batch_face_FP, batch_face_FN = calculate_metrics(pred_head_contacts, gt_head_contact, threshold=0.5)
            batch_TP += batch_face_TP
            batch_TN += batch_face_TN
            batch_FP += batch_face_FP
            batch_FN += batch_face_FN

            batch_hand_TP, batch_hand_TN, batch_hand_FP, batch_hand_FN = calculate_metrics(pred_hand_contacts, gt_hand_contact, threshold=0.5)
            batch_TP += batch_hand_TP
            batch_TN += batch_hand_TN
            batch_FP += batch_hand_FP
            batch_FN += batch_hand_FN

            global_TP += batch_TP
            global_TN += batch_TN
            global_FP += batch_FP
            global_FN += batch_FN

            global_hand_TP += batch_hand_TP
            global_hand_TN += batch_hand_TN
            global_hand_FP += batch_hand_FP
            global_hand_FN += batch_hand_FN

            global_face_TP += batch_face_TP
            global_face_TN += batch_face_TN
            global_face_FP += batch_face_FP
            global_face_FN += batch_face_FN

            batch_presence_TP, batch_presence_TN, batch_presence_FP, batch_presence_FN = calculate_metrics(pred_contact_presence, gt_contact_presence, threshold=0.5)
            global_presence_TP += batch_presence_TP
            global_presence_TN += batch_presence_TN
            global_presence_FP += batch_presence_FP
            global_presence_FN += batch_presence_FN
            
            if len(error_deform)>0:
                DeformError.update(np.mean(error_deform), batch_size )
            if len(error_deform_large)>0:
                LargeDeformError.update(np.mean(error_deform_large), error_deform_large.shape[0])
            if len(error_vertices_trans)>0:
                mPVE_trans.update(np.mean(error_vertices_trans), batch_size )
            if len(error_joints)>0:
                mPJPE.update(np.mean(error_joints), batch_size )
            if len(error_joints_pa)>0:
                PAmPJPE.update(np.mean(error_joints_pa), batch_size )

            if batch_gt_touch_count > 0:
                Touchness.update( (batch_gt_touch_count-batch_non_touching_count) / batch_gt_touch_count, batch_gt_touch_count)
            if batch_count > 0:
                NonColRatio.update(batch_non_col_count / batch_count, batch_count)
                ColDist.update(batch_collision_distance_sum / batch_count, batch_count)

    gt_kp_imgs = np.vstack(gt_kp_imgs)
    pred_kp_imgs = np.vstack(pred_kp_imgs)
    gt_vs_imgs = np.vstack(gt_vs_imgs)
    pred_vs_imgs = np.vstack(pred_vs_imgs)
    pred_kp_parametric_imgs = np.vstack(pred_kp_parametric_imgs)
    pred_vs_parametric_imgs = np.vstack(pred_vs_parametric_imgs)
    render_imgs = np.vstack(render_imgs)
    visual_imgs = np.hstack([gt_kp_imgs, pred_kp_imgs, pred_kp_parametric_imgs, gt_vs_imgs, pred_vs_imgs, pred_vs_parametric_imgs, render_imgs])


    if args.local_rank == 0:
        stamp = str(epoch)
        temp_fname = args.output_dir + 'val_visual_' + stamp + '.jpg'
        cv2.imwrite(temp_fname, visual_imgs)

    TP = all_gather(int(global_TP))
    TP = sum(TP)
    TN = all_gather(int(global_TN))
    TN = sum(TN)
    FP = all_gather(int(global_FP))
    FP = sum(FP)
    FN = all_gather(int(global_FN))
    FN = sum(FN)
    all_precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    all_recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    all_accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
    all_f1_score = 2 * all_precision * all_recall / (all_precision + all_recall) if (all_precision + all_recall) != 0 else 0

    hand_TP = all_gather(int(global_hand_TP))
    hand_TP = sum(hand_TP)
    hand_TN = all_gather(int(global_hand_TN))
    hand_TN = sum(hand_TN)
    hand_FP = all_gather(int(global_hand_FP))
    hand_FP = sum(hand_FP)
    hand_FN = all_gather(int(global_hand_FN))
    hand_FN = sum(hand_FN)
    hand_precision = hand_TP / (hand_TP + hand_FP) if (hand_TP + hand_FP) != 0 else 0
    hand_recall = hand_TP / (hand_TP + hand_FN) if (hand_TP + hand_FN) != 0 else 0
    hand_accuracy = (hand_TP + hand_TN) / (hand_TP + hand_TN + hand_FP + hand_FN) if (hand_TP + hand_TN + hand_FP + hand_FN) != 0 else 0
    hand_f1_score = 2 * hand_precision * hand_recall / (hand_precision + hand_recall) if (hand_precision + hand_recall) != 0 else 0

    face_TP = all_gather(int(global_face_TP))
    face_TP = sum(face_TP)
    face_TN = all_gather(int(global_face_TN))
    face_TN = sum(face_TN)
    face_FP = all_gather(int(global_face_FP))
    face_FP = sum(face_FP)
    face_FN = all_gather(int(global_face_FN))
    face_FN = sum(face_FN)
    face_precision = face_TP / (face_TP + face_FP) if (face_TP + face_FP) != 0 else 0
    face_recall = face_TP / (face_TP + face_FN) if (face_TP + face_FN) != 0 else 0
    face_accuracy = (face_TP + face_TN) / (face_TP + face_TN + face_FP + face_FN) if (face_TP + face_TN + face_FP + face_FN) != 0 else 0
    face_f1_score = 2 * face_precision * face_recall / (face_precision + face_recall) if (face_precision + face_recall) != 0 else 0

    presence_TP = all_gather(int(global_presence_TP))
    presence_TP = sum(presence_TP)
    presence_TN = all_gather(int(global_presence_TN))
    presence_TN = sum(presence_TN)
    presence_FP = all_gather(int(global_presence_FP))
    presence_FP = sum(presence_FP)
    presence_FN = all_gather(int(global_presence_FN))
    presence_FN = sum(presence_FN)
    presence_precision = presence_TP / (presence_TP + presence_FP) if (presence_TP + presence_FP) != 0 else 0
    presence_recall = presence_TP / (presence_TP + presence_FN) if (presence_TP + presence_FN) != 0 else 0
    presence_accuracy = (presence_TP + presence_TN) / (presence_TP + presence_TN + presence_FP + presence_FN) if (presence_TP + presence_TN + presence_FP + presence_FN) != 0 else 0
    presence_f1_score = 2 * presence_precision * presence_recall / (presence_precision + presence_recall) if (presence_precision + presence_recall) != 0 else 0

    val_DeformError = all_gather(float(DeformError.avg))
    val_DeformError = sum(val_DeformError)/len(val_DeformError)

    LargeDeformError_sum = all_gather(float(LargeDeformError.sum))
    LargeDeformError_count = all_gather(float(LargeDeformError.count))
    val_LargeDeformError = sum(LargeDeformError_sum)/sum(LargeDeformError_count)

    val_mPVE = all_gather(float(mPVE.avg))
    val_mPVE = sum(val_mPVE)/len(val_mPVE)

    val_mPVE_trans = all_gather(float(mPVE_trans.avg))
    val_mPVE_trans = sum(val_mPVE_trans)/len(val_mPVE_trans)

    val_mPJPE = all_gather(float(mPJPE.avg))
    val_mPJPE = sum(val_mPJPE)/len(val_mPJPE)

    val_PAmPJPE = all_gather(float(PAmPJPE.avg))
    val_PAmPJPE = sum(val_PAmPJPE)/len(val_PAmPJPE)

    val_count = all_gather(float(mPVE.count))
    val_count = sum(val_count)

    val_Touchness = all_gather(float(Touchness.avg))
    val_Touchness = sum(val_Touchness)/len(val_Touchness)

    val_NonColRatio = all_gather(float(NonColRatio.avg))
    val_NonColRatio = sum(val_NonColRatio)/len(val_NonColRatio)

    val_ColDist = all_gather(float(ColDist.avg))
    val_ColDist = sum(val_ColDist)/len(val_ColDist)


    return val_mPVE, val_mPVE_trans, val_mPJPE, val_PAmPJPE, val_count, val_Touchness, val_NonColRatio, val_ColDist, all_precision, all_recall, all_accuracy, all_f1_score, hand_precision, hand_recall, hand_accuracy, hand_f1_score, face_precision, face_recall, face_accuracy, face_f1_score, presence_precision, presence_recall, presence_accuracy, presence_f1_score, val_DeformError, val_LargeDeformError


def visualize_mesh( renderer,
                    images,
                    gt_keypoints_2d,
                    pred_vertices, 
                    pred_camera,
                    pred_keypoints_2d):
    """Tensorboard logging."""
    gt_keypoints_2d = gt_keypoints_2d.cpu().numpy()
    to_lsp = list(range(14))
    rend_imgs = []
    batch_size = pred_vertices.shape[0]
    # Do visualization for the first 6 images of the batch
    for i in range(min(batch_size, 10)):
        img = images[i].cpu().numpy().transpose(1,2,0)
        # Get LSP keypoints from the full list of keypoints
        gt_keypoints_2d_ = gt_keypoints_2d[i, to_lsp]
        pred_keypoints_2d_ = pred_keypoints_2d.cpu().numpy()[i, to_lsp]
        # Get predict vertices for the particular example
        vertices = pred_vertices[i].cpu().numpy()
        cam = pred_camera[i].cpu().numpy()
        # Visualize reconstruction and detected pose
        rend_img = visualize_reconstruction(img, 224, gt_keypoints_2d_, vertices, pred_keypoints_2d_, cam, renderer)
        rend_img = rend_img.transpose(2,0,1)
        rend_imgs.append(torch.from_numpy(rend_img))   
    rend_imgs = make_grid(rend_imgs, nrow=1)
    return rend_imgs

def visualize_mesh_test( renderer,
                    images,
                    gt_keypoints_2d,
                    pred_vertices, 
                    pred_camera,
                    pred_keypoints_2d,
                    PAmPJPE_h36m_j14):
    """Tensorboard logging."""
    gt_keypoints_2d = gt_keypoints_2d.cpu().numpy()
    to_lsp = list(range(14))
    rend_imgs = []
    batch_size = pred_vertices.shape[0]
    # Do visualization for the first 6 images of the batch
    for i in range(min(batch_size, 10)):
        img = images[i].cpu().numpy().transpose(1,2,0)
        # Get LSP keypoints from the full list of keypoints
        gt_keypoints_2d_ = gt_keypoints_2d[i, to_lsp]
        pred_keypoints_2d_ = pred_keypoints_2d.cpu().numpy()[i, to_lsp]
        # Get predict vertices for the particular example
        vertices = pred_vertices[i].cpu().numpy()
        cam = pred_camera[i].cpu().numpy()
        score = PAmPJPE_h36m_j14[i]
        # Visualize reconstruction and detected pose
        rend_img = visualize_reconstruction_test(img, 224, gt_keypoints_2d_, vertices, pred_keypoints_2d_, cam, renderer, score)
        rend_img = rend_img.transpose(2,0,1)
        rend_imgs.append(torch.from_numpy(rend_img))   
    rend_imgs = make_grid(rend_imgs, nrow=1)
    return rend_imgs


def parse_args():
    parser = argparse.ArgumentParser()
    #########################################################
    # Data related arguments
    #########################################################
    parser.add_argument("--data_dir", default='datasets', type=str, required=False,
                        help="Directory with all datasets, each in one subfolder")
    parser.add_argument("--train_yaml", default='imagenet2012/train.yaml', type=str, required=False,
                        help="Yaml file with all data for training.")
    parser.add_argument("--val_yaml", default='imagenet2012/test.yaml', type=str, required=False,
                        help="Yaml file with all data for validation.")
    parser.add_argument("--num_workers", default=4, type=int, 
                        help="Workers in dataloader.")
    parser.add_argument("--img_scale_factor", default=1, type=int, 
                        help="adjust image resolution.") 
    #########################################################
    # Loading/saving checkpoints
    #########################################################
    parser.add_argument("--model_name_or_path", default='src/modeling/bert/bert-base-uncased/', type=str, required=False,
                        help="Path to pre-trained transformer model or model type.")
    parser.add_argument("--resume_checkpoint", default=None, type=str, required=False,
                        help="Path to specific checkpoint for resume training.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--config_name", default="", type=str, 
                        help="Pretrained config name or path if not the same as model_name.")
    #########################################################
    # Training parameters
    #########################################################
    parser.add_argument("--per_gpu_train_batch_size", default=30, type=int, 
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=30, type=int, 
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--lr', "--learning_rate", default=1e-4, type=float, 
                        help="The initial lr.")
    parser.add_argument("--num_train_epochs", default=200, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument("--vertices_loss_weight", default=1000.0, type=float)          
    parser.add_argument("--joints_loss_weight_3d", default=1000.0, type=float)
    parser.add_argument("--joints_loss_weight_2d", default=500.0, type=float)
    parser.add_argument("--contacts_loss_weight", default=300.0, type=float)
    parser.add_argument("--params_loss_weight", default=500.0, type=float)
    parser.add_argument('--collision_loss_weight', type=float, default=500.0)
    parser.add_argument('--touch_loss_weight', type=float, default=100.0)
    parser.add_argument('--normal_vector_loss_weight', type=float, default=0.0)
    parser.add_argument('--edge_length_loss_weight', type=float, default=0.0)
    parser.add_argument('--presence_loss_weight', type=float, default=500.0)
    parser.add_argument("--vloss_head_full", default=0.2, type=float)
    parser.add_argument("--vloss_head_sub", default=0.2, type=float)
    parser.add_argument("--vloss_head_sub2", default=0.2, type=float)
    parser.add_argument("--vloss_hand_full", default=0.2, type=float)
    parser.add_argument("--vloss_hand_sub", default=0.2, type=float)
    parser.add_argument("--closs_hand_full", default=0.2, type=float)
    parser.add_argument("--closs_hand_sub", default=0.2, type=float)
    parser.add_argument("--closs_head_full", default=0.2, type=float)
    parser.add_argument("--closs_head_sub", default=0.2, type=float)
    parser.add_argument("--closs_head_sub2", default=0.2, type=float)
    parser.add_argument("--drop_out", default=0.1, type=float, 
                        help="Drop out ratio in BERT.")
    #########################################################
    # Model architectures
    #########################################################
    parser.add_argument('-a', '--arch', default='hrnet-w64',
                    help='CNN backbone architecture: hrnet-w64, hrnet, resnet50')
    parser.add_argument("--num_hidden_layers", default=4, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--hidden_size", default=-1, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--num_attention_heads", default=4, type=int, required=False, 
                        help="Update model config if given. Note that the division of "
                        "hidden_size / num_attention_heads should be in integer.")
    parser.add_argument("--intermediate_size", default=-1, type=int, required=False, 
                        help="Update model config if given.")
    parser.add_argument("--input_feat_dim", default='2051,512,128', type=str, 
                        help="The Image Feature Dimension.")          
    parser.add_argument("--hidden_feat_dim", default='1024,256,128', type=str, 
                        help="The Image Feature Dimension.")   
    parser.add_argument("--output_feat_dim", default='512,128,3', type=str, 
                        help="The Image Feature Dimension.")                       
    parser.add_argument("--legacy_setting", default=True, action='store_true',)
    #########################################################
    # Others
    #########################################################
    parser.add_argument("--run_eval_only", default=False, action='store_true',) 
    parser.add_argument('--logging_steps', type=int, default=500, 
                        help="Log every X steps.")
    parser.add_argument("--device", type=str, default='cuda', 
                        help="cuda or cpu")
    parser.add_argument('--seed', type=int, default=88, 
                        help="random seed for initialization.")
    parser.add_argument("--local_rank", type=int, default=0, 
                        help="For distributed training.")
    parser.add_argument("--model_dim_1", default=512, type=int)
    parser.add_argument("--model_dim_2", default=128, type=int)
    parser.add_argument("--feedforward_dim_1", default=2048, type=int)
    parser.add_argument("--feedforward_dim_2", default=512, type=int)
    parser.add_argument("--conv_1x1_dim", default=2048, type=int)
    parser.add_argument("--transformer_dropout", default=0.1, type=float)
    parser.add_argument("--transformer_nhead", default=8, type=int)
    parser.add_argument("--pos_type", default='sine', type=str)
    parser.add_argument("--edge_length_loss", default="false", type=str)
    parser.add_argument("--normal_vector_loss", default="false", type=str)

    parser.add_argument('--win_size', type=int, default=0)
    parser.add_argument('--save', type=int, default=0)
    parser.add_argument('--cam_space_deform', type=int, default=1) 
    parser.add_argument('--back_aug', type=int, default=1)
    parser.add_argument('--train_imgrot_aug', type=int, default=1)
    parser.add_argument('--img_wh', type=tuple, default=(1920,1080))
    parser.add_argument('--max_epoch', type=int, default=1500)
    parser.add_argument('--n_pca', type=int, default=45)
    parser.add_argument('--pre_train', type=int, default=199)
    parser.add_argument('--dist_thresh', type=float, default=0.1)
    parser.add_argument('--hidden', type=int, default=5023*1)
    parser.add_argument('--dyn_iter', type=int, default=200)#args.num_workers
    parser.add_argument('--deform_thresh', type=int, default=0)
    parser.add_argument('--flipping', type=int, default=1)
    parser.add_argument('--debug_val', type=str, default="false")
    parser.add_argument('--data_path', default='/code/datasets/DecafDataset/', type=str)
    parser.add_argument('--image_data_path', type=str, default="/code/datasets/DecafDataset_images/") 
    parser.add_argument('--single_image_path', type=str, default="/code/datasets/Decaf_imgs_single/")
    parser.add_argument('--deform_reg_weight', type=float, default=10.0)
    parser.add_argument('--deformation_loss_weight', type=float, default=5000.0)
    parser.add_argument('--inthewild_deformation_loss_weight', type=float, default=1000.0)
    parser.add_argument('--depth_loss_weight', type=float, default=100.0)
    parser.add_argument('--discriminator_loss_weight', type=float, default=100.0)
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--decaf_multiscale', action='store_true')
    parser.add_argument('--scale_range', type=str, default="0.75,1.25")
    parser.add_argument('--burn_in_epochs', type=int, default=0)
    parser.add_argument('--itw_resample', type=int, default=50)
    parser.add_argument('--inthewild_root_dir', type=str, default="/code/datasets/itw_new_crop_center_clean")
    parser.add_argument('--inthewild_image_num', type=int, default=None)
    parser.add_argument('--self_supervision', default=False, action='store_true',) 
    args = parser.parse_args()
    return args


def main(args):
    global logger
    # Setup CUDA, GPU & distributed training
    args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = args.num_gpus > 1
    args.device = torch.device(args.device)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://'
        )
        local_rank = int(os.environ["LOCAL_RANK"])
        args.device = torch.device("cuda", local_rank)
        synchronize()

    mkdir(args.output_dir)
    logger = setup_logger("DICE", args.output_dir, get_rank())
    set_seed(args.seed, args.num_gpus)
    logger.info("Using {} GPUs".format(args.num_gpus))

    ### path setup ###
    asset_path = args.data_path+"/assets/"
    mano_model_path = asset_path+'/mano_v1_2/models/MANO_RIGHT.pkl'
    flame_model_path = asset_path+"/generic_model.pkl"
    flame_landmark_path = asset_path+"/landmark_embedding.npy"

    mano_model = MANO().to(args.device)
    mano_model.layer = mano_model.layer.cuda()
    rh_model = mano.model.load(
              model_path=mano_model_path,
              is_right= True, 
              num_pca_comps=args.n_pca, 
              batch_size=1, 
              flat_hand_mean=True).to(args.device)

    flame_model = FLAME(flame_model_path, flame_landmark_path).to(args.device)
    flame_faces = get_FLAME_faces(flame_model_path)

    transforms = torch.load("head_mesh_transforms.pt")
    A, U, D, F = transforms["A"], transforms["U"], transforms["D"], transforms["F"]
    head_F = [f.numpy() for f in F]
    head_sampler = Mesh(A, U, D, num_downsampling=2, nsize=1)

    transforms = torch.load("hand_mesh_transforms.pt")
    A, U, D, F = transforms["A"], transforms["U"], transforms["D"], transforms["F"]
    hand_F = [f.numpy() for f in F]
    hand_sampler = Mesh(A, U, D, num_downsampling=1, nsize=1)

    # Renderer for visualization
    hand_renderer = Renderer(faces=rh_model.faces)
    face_renderer = Renderer(faces=flame_faces.cpu().numpy())

    trans_encoder = []

    input_feat_dim = [int(item) for item in args.input_feat_dim.split(',')]
    hidden_feat_dim = [int(item) for item in args.hidden_feat_dim.split(',')]
    output_feat_dim = [int(item) for item in args.output_feat_dim.split(',')]
    
    if args.run_eval_only==True and args.resume_checkpoint!=None and args.resume_checkpoint!='None' and 'state_dict' not in args.resume_checkpoint:
        # if only run eval, load checkpoint
        logger.info("Evaluation: Loading from checkpoint {}".format(args.resume_checkpoint))
        _dice_network = torch.load(args.resume_checkpoint)
    else:
        for i in range(len(output_feat_dim)):
            config_class, model_class = BertConfig, DICE_Module
            config = config_class.from_pretrained(args.config_name if args.config_name \
                    else args.model_name_or_path)

            config.output_attentions = False
            config.hidden_dropout_prob = args.drop_out
            config.img_feature_dim = input_feat_dim[i] 
            config.output_feature_dim = output_feat_dim[i]
            args.hidden_size = hidden_feat_dim[i]

            if args.legacy_setting==True:
                args.intermediate_size = -1
            else:
                args.intermediate_size = int(args.hidden_size*4)

            update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']

            for idx, param in enumerate(update_params):
                arg_param = getattr(args, param)
                config_param = getattr(config, param)
                if arg_param > 0 and arg_param != config_param:
                    logger.info("Update config parameter {}: {} -> {}".format(param, config_param, arg_param))
                    setattr(config, param, arg_param)

            assert config.hidden_size % config.num_attention_heads == 0
            model = model_class(config=config) 
            logger.info("Init model from scratch.")
            trans_encoder.append(model)

        
        deform_encoder = []
        input_feat_dim = [512, 128]
        hidden_feat_dim = [256, 64]
        output_feat_dim = input_feat_dim[1:] + [4]

        for i in range(len(output_feat_dim)):
            config_class, model_class = BertConfig, DICE_Module
            config = config_class.from_pretrained(args.config_name if args.config_name \
                    else args.model_name_or_path)

            config.output_attentions = False
            config.hidden_dropout_prob = args.drop_out
            config.img_feature_dim = input_feat_dim[i] 
            config.output_feature_dim = output_feat_dim[i]
            args.hidden_size = hidden_feat_dim[i]

            if args.legacy_setting==True:
                args.intermediate_size = -1
            else:
                args.intermediate_size = int(args.hidden_size*4)

            update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']

            for idx, param in enumerate(update_params):
                arg_param = getattr(args, param)
                config_param = getattr(config, param)
                if arg_param > 0 and arg_param != config_param:
                    logger.info("Update config parameter {}: {} -> {}".format(param, config_param, arg_param))
                    setattr(config, param, arg_param)

            assert config.hidden_size % config.num_attention_heads == 0
            model = model_class(config=config) 
            logger.info("Init model from scratch.")
            deform_encoder.append(model)

        if args.arch=='hrnet-w64':
            hrnet_yaml = 'models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = 'models/hrnet/hrnetv2_w64_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
            logger.info('=> loading hrnet-v2-w64 model')
        elif 'efficientnet' in args.arch:
            print(f"loading pretrained {args.arch}")
            backbone = EfficientNet.from_pretrained(args.arch)
        else:
            print("=> using pre-trained model '{}'".format(args.arch))
            backbone = models.__dict__[args.arch](pretrained=True)
            # remove the last fc layer
            backbone = torch.nn.Sequential(*list(backbone.children())[:-2])

        trans_encoder = torch.nn.Sequential(OrderedDict([
            ('layer1', trans_encoder[0]),
            ('layer2', trans_encoder[1]),
            ('layer3', trans_encoder[2]),
        ]))
        deform_encoder = torch.nn.Sequential(OrderedDict([
            ('layer1', deform_encoder[0]),
            ('layer2', deform_encoder[1]),
        ]))
        backbone_total_params = sum(p.numel() for p in backbone.parameters())
        logger.info('Backbone total parameters: {}'.format(backbone_total_params))

        config.output_attentions = False

        _dice_network = DICE_Network(args, config, backbone, trans_encoder, deform_encoder)
        hand_discriminator = Hand_Discriminator()
        face_discriminator = Face_Discriminator()

        transformer_total_params = sum([params.numel() for name, params in _dice_network.named_parameters() if 'backbone' not in name])
        print(transformer_total_params)

        if args.resume_checkpoint!=None and args.resume_checkpoint!='None':
            logger.info("Loading state dict from checkpoint {}".format(args.resume_checkpoint))
            cpu_device = torch.device('cpu')
            state_dict = torch.load(args.resume_checkpoint, map_location=cpu_device)
            hand_state_dict_path = args.resume_checkpoint.replace("state_dict", "hand_d_state_dict")
            face_state_dict_path = args.resume_checkpoint.replace("state_dict", "face_d_state_dict")
            _dice_network.load_state_dict(state_dict, strict=False)
            del state_dict
            if os.path.exists(hand_state_dict_path) and os.path.exists(face_state_dict_path):
                hand_state_dict = torch.load(hand_state_dict_path, map_location=cpu_device)
                face_state_dict = torch.load(face_state_dict_path, map_location=cpu_device)
                print("loading hand and face discriminator state dict")
                hand_discriminator.load_state_dict(hand_state_dict, strict=False)
                face_discriminator.load_state_dict(face_state_dict, strict=False)
                del hand_state_dict
                del face_state_dict
            
    
    _dice_network.to(args.device)
    hand_discriminator.to(args.device)
    face_discriminator.to(args.device)
    logger.info("Training parameters %s", args)

    print("discriminator for all data")


    if args.run_eval_only==True:
        val_dataloader = make_decaf_and_inthewild_data_loader(args, args.distributed, is_train=False)
        run_eval_general(args, val_dataloader, _dice_network, rh_model, flame_model, hand_mesh_sampler)

    else:
        if args.dataset == "combined" or args.dataset == "inthewild":
            train_dataloader = make_decaf_and_inthewild_data_loader(args, args.distributed, is_train=True)
            val_dataloader = make_decaf_data_loader(args, args.distributed, is_train=False)
        elif args.dataset == "decaf":
            train_dataloader = make_decaf_data_loader(args, args.distributed, is_train=True)
            val_dataloader = make_decaf_data_loader(args, args.distributed, is_train=False)
        else:
            raise ValueError("Please specify a dataset to train on")

        run(args, train_dataloader, val_dataloader, _dice_network, mano_model=mano_model, hand_model=rh_model, head_model=flame_model, hand_sampler=hand_sampler, head_sampler=head_sampler, hand_renderer=hand_renderer, face_renderer=face_renderer, hand_F=hand_F, head_F=head_F, hand_discriminator=hand_discriminator, face_discriminator=face_discriminator)


if __name__ == "__main__":
    args = parse_args()
    main(args)
