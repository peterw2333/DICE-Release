python setup.py build develop
exp_name="example_train"
mkdir output
mkdir output/$exp_name/
export WANDB_MODE="dryrun"
cp src/tools/dice.py output/$exp_name/
cp src/modeling/bert/modeling.py output/$exp_name/
CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.launch --nproc_per_node=1 --master_port 47786  \
       src/tools/dice.py \
       --arch hrnet-w64 \
       --num_workers 24 \
       --per_gpu_train_batch_size 16 \
       --per_gpu_eval_batch_size 16 \
       --lr 3e-5 \
       --num_train_epochs 40 \
       --logging_steps 100 \
       --output_dir output/$exp_name/ \
       --edge_length_loss "false" \
       --normal_vector_loss "false" \
       --vertices_loss_weight 1000 \
       --joints_loss_weight_2d 250 \
       --joints_loss_weight_3d 500 \
       --deformation_loss_weight 3000 \
       --inthewild_deformation_loss_weight 200 \
       --collision_loss_weight 500 \
       --touch_loss_weight 100 \
       --contacts_loss_weight 300 \
       --normal_vector_loss_weight 0 \
       --edge_length_loss_weight 0 \
       --params_loss_weight 500 \
       --deform_reg_weight 4.0 \
       --run_name $exp_name \
       --input_feat_dim "2051,512,128" \
       --hidden_feat_dim "1024,256,128" \
       --output_feat_dim "512,128,3" \
       --dataset "combined" \
       --discriminator_loss_weight 20 \
       --burn_in_epochs 2 \
       --depth_loss_weight 50 \
       --itw_resample 50 \
       --inthewild_root_dir /code/datasets/inthewild_dataset_centered \
       --inthewild_image_num 100
