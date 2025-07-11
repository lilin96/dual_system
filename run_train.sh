# lr=3e-4
# wd=5e-3
# dense_interpolation=1
# interpolation_length=20
# num_history=1  
# diffusion_timesteps=25
# B=15 
# C=192
ngpus=1 
# backbone=clip
# image_size="256,256"
# relative_action=1
# fps_subsampling_factor=3
# lang_enhanced=1
# gripper_loc_bounds=tasks/calvin_rel_traj_location_bounds_task_ABC_D.json
# gripper_buffer=0.01
# val_freq=500 
# quaternion_format=wxyz
# train_iters=67000
# stage2_train_iters=100000
# training_checkpoint=/3d_diffuser_actor/train_logs/diffuser_actor_calvin_nohistory.pth

# run_log_dir=OpenHelix_ABC_D-gpu$ngpus-step1$train_iters-step2$stage2_train_iters-C$C-B$B-lr$lr-DI$dense_interpolation-$interpolation_length-H$num_history-DT$diffusion_timesteps-backbone$backbone-S$image_size-R$relative_action-wd$wd

export PYTHONPATH=`pwd`:$PYTHONPATH

CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node $ngpus --master_port 12355 \
    main_trajectory_act_simple.py 