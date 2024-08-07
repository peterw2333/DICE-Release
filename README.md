# DICE

## Note
The [placeholder]s refers to the files that haven't been publicly released anywhere else. We will release these files  upon publishing of the paper.

## Usage
Examples of training and inference script has been provided in the root folder: `train.sh` and `infer.sh`. 

## Environment Preparation
Install Pytorch3D by running: `git clone https://github.com/facebookresearch/pytorch3d.git&&cd ./pytorch3d&&git checkout tags/v0.7.2&&pip install -e .&&cd ..`

install the other required packages via `pip install -r requirements.txt`.

## Data preparation
### Decaf
- Download the main dataset from [DecafDatasetScript](https://github.com/soshishimada/DecafDatasetScript).
- Then, use `src/decaf/image_cropper.py` to crop the videos into images and bounding box files.
- Also, use `src/decaf/image_cropper_single.py` to crop the videos into single images, for our model input.

### Motion Datasets
- Prepare the Freihand dataset according to this [instruction](https://github.com/postech-ami/FastMETRO/blob/main/docs/Download.md)
- Download the RenderMe-360 FLAME parameters from [placeholder].

### In-the-wild dataset

The in-the-wild dataset need to be structured like this:
```
.
├── depth_maps
├── face_keypoints
├── hand_keypoints
└── images
```
- The `images` folder contain the relevant images.
- The `hand_keypoint` folder contain the `.npy` files storing the hand keypoints, with the same name of the image. E.g. `1.jpg` corresponds to `1.npy`.
- The `face_keypoint` folder contain the `.npy` files storing the face keypoints, with the same name of the image. E.g. `1.jpg` corresponds to `1.npy`.
- The `depth_map` folder contain the `.npy` files of the depth maps, with the same name of the image. E.g. `1.jpg` corresponds to `1.npy`.
To prepare the depth map, refer to [Marigold](https://github.com/prs-eth/Marigold), using the LCM model and save the npy output.

### Dependency Files
run `sh download_models.sh` in the root folder to download the pretrained HRNet-W64 checkpoint.
- create the folder `src/common/utils/human_model_files` and download the relevant files according to this [instruction](https://github.com/IDEA-Research/OSX?tab=readme-ov-file#4-directory).
- Download `head_mesh_transforms.pt` and `hand_mesh_transforms.pt` [placeholder] and save to the root folder.
- Download `head_ref_vs.pt`, `rh_ref_vs.pt`, and `stiffness_final.npy` [placeholder] and save to `src/modeling/data/`.
- Download `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` from [SMPLify](http://smplify.is.tue.mpg.de/), and place it at `src/modeling/data`.
- Download `MANO_RIGHT.pkl` from [MANO](https://mano.is.tue.mpg.de/), and place it at `src/modeling/data`.


## Acknowledgements
Our implementation and experiments are built on top of open-source GitHub repositories. We thank all the authors who made their code public, which tremendously accelerates our project progress. 

[HRNet/HRNet-Image-Classification](https://github.com/HRNet/HRNet-Image-Classification)

[huggingface/transformers](https://github.com/huggingface/transformers)

[microsoft/MeshTransformer](https://github.com/microsoft/MeshTransformer)

[MPI-IS/mesh](https://github.com/MPI-IS/mesh)

[facebookresearch/pytorch3d](https://github.com/facebookresearch/pytorch3d)