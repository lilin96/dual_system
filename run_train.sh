lr=5e-5
# lr=0.0003
# wd=5e-3
# dense_interpolation=1
# interpolation_length=20
# num_history=1  
# diffusion_timesteps=25
# B=15 
# C=192
batch_size=16
ngpus=4
# backbone=clip
# image_size="256,256"
# relative_action=1
# fps_subsampling_factor=3
# lang_enhanced=1
# gripper_loc_bounds=tasks/calvin_rel_traj_location_bounds_task_ABC_D.json
# gripper_buffer=0.01
# val_freq=500 
# quaternion_format=wxyz
run_log_dir="run_lr_$lr"
train_iters=670
stage2_train_iters=10_000
llava_dir='/home/users/ntu/yongxias/scratch/lilin_projects/pretrained/LLaVA-Lightning-7B-delta-v1-1'
vision_tower='/home/users/ntu/yongxias/scratch/lilin_projects/pretrained/clip-vit-large-patch14'
training_checkpoint='/home/users/ntu/yongxias/scratch/lilin_projects/pretrained/TCP/tcp_b2d.ckpt'
# LCB_checkpoint='/home/users/ntu/yongxias/scratch/lilin_projects/dual_system/train_logs/exp/run/0009999/pytorch_model.bin'

# run_log_dir=OpenHelix_ABC_D-gpu$ngpus-step1$train_iters-step2$stage2_train_iters-C$C-B$B-lr$lr-DI$dense_interpolation-$interpolation_length-H$num_history-DT$diffusion_timesteps-backbone$backbone-S$image_size-R$relative_action-wd$wd

export PYTHONPATH=`pwd`:$PYTHONPATH

CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node $ngpus --master_port 12355 \
    main_trajectory_act_simple.py \
    --lr $lr \
    --batch_size $batch_size\
    --llava_dir $llava_dir \
    --vision_tower $vision_tower \
    --training_checkpoint $training_checkpoint\
    --stage2_train_iters $stage2_train_iters\
    --train_iters $train_iters\
    --run_log_dir $run_log_dir \
    --eval_only 0 >> "runtrain.log" 2>&1 &