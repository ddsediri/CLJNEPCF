#!/bin/bash

pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

python ./FeatureEncoderDriver.py --trainset="./Dataset/Train" --shapes_list_file="train.txt" --patches_per_shape=8192 --points_per_patch=500 --patch_radius=0.05 --batchSize=512 --lr=3e-4 --wd=1e-4 --workers=24 --nepochs=150 --num_noise_levels=6 --device_id=0

python ./RegressorDriver.py --checkpoint_path="./FeatureEncoderTrained/chkpt_cbs_512_ep150.pth.tar" --trainset="./Dataset/Train" --shapes_list_file='train.txt' --batchSize=12 --patches_per_shape=8192 --points_per_patch=500 --patch_radius=0.05 --lr=1e-2 --workers=24 --nepochs=30 --num_noise_levels=6 --train_type="regression" --upstream_cbs=512 --downstream_alpha=0.9 --downstream_beta=0.01 --downstream_delta=0.3 --downstream_gamma=12 --device_id=0 

python ./Inference.py --checkpoint_path="./RegressorTrained/chkpt_cbs_512_ep30_a0.90_b0.01_d0.30_g12.pth.tar" --shapes_list_file="test.txt" --eval_iter_nums=4
