#!/bin/bash

pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

python ./Inference.py --checkpoint_path="./RegressorPreTrained/chkpt_cbs_512_ep30_a0.90_b0.01_d0.30_g12.pth.tar" --shapes_list_file="test.txt" --eval_iter_nums=4
