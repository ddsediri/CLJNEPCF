#!/bin/bash

pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

python ./FeatureEncoderDriver.py --trainset="./Dataset/Train" --shapes_list_file="train.txt" --patches_per_shape=8192 --points_per_patch=500 --patch_radius=0.05 --batchSize=512 --lr=3e-4 --wd=1e-4 --workers=24 --nepochs=150 --num_noise_levels=6 --device_id=0
