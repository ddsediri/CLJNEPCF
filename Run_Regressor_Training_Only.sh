#!/bin/bash

pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

python ./RegressorDriver.py --checkpoint_path="./FeatureEncoderPreTrained/chkpt_cbs_512_ep150.pth.tar" --trainset="./Dataset/Train" --shapes_list_file='train.txt' --batchSize=12 --patches_per_shape=8192 --points_per_patch=500 --patch_radius=0.05 --lr=1e-2 --workers=24 --nepochs=30 --num_noise_levels=6 --train_type="regression" --upstream_cbs=512 --downstream_alpha=0.9 --downstream_beta=0.01 --downstream_delta=0.3 --downstream_gamma=12 --device_id=0 
