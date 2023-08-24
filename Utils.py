import os
import shutil

from sklearn.decomposition import PCA
import math
import torch
import argparse

import numpy as np
import torch
import yaml

##########################Parameters########################
#
#
#
#
###############################################################

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments():
    parser = argparse.ArgumentParser()
    # naming / file handling
    parser.add_argument('--name', type=str, default='pcdenoising', help='training run name')
    parser.add_argument('--checkpoint_path', type=str, default='', help='The name of the checkpoint being loaded (feature encoder checkpoint for regressor training or regressor checkpoint for inference)')
    parser.add_argument('--trainset', type=str, default='./Dataset/Train', help='training set file name')
    parser.add_argument('--testset', type=str, default='./Dataset/Test', help='testing set file name')
    parser.add_argument('--save_dir', type=str, default='./Results', help='')
    parser.add_argument('--shape_name', type=str, default='', help='')
    parser.add_argument('--shapes_list_file', type=str, default='train.txt', help='.txt file containing shape names to train or test on')

    # training parameters
    parser.add_argument('--nepochs', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--manualSeed', type=int, default=3627473, help='manual seed')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--patches_per_shape', type=int, default=8192, help='number of patches taken from a shape during each training epoch')
    parser.add_argument('--points_per_patch', type=int, default=500, help='number of points within a patch')
    parser.add_argument('--patch_radius', type=float, default=0.05, help='radius of a patch')
    parser.add_argument('--num_noise_levels', type=int, default=2, help='The number of levels of noise for noisy shapes')

    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='gradient descent momentum')
    parser.add_argument('--model_interval', type=int, default=5, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--train_type', type=str, default='clearning', help='Contrastive learning or normal estimation?')

    # others parameters
    parser.add_argument('--upstream_cbs', type=int, default='512', help='contrastive learning batch size')
    parser.add_argument('--downstream_alpha', type=float, default='0.9', help='alpha parameter: weight for normal/position loss contribution')
    parser.add_argument('--downstream_beta', type=float, default='0.01', help='beta parameter: weight for position loss regularization')
    parser.add_argument('--downstream_delta', type=float, default='0.01', help='delta parameter: to control shape of normal loss function')
    parser.add_argument('--downstream_gamma', type=int, default='12', help='gamma parameter: to control cosine similarity penalization')
    parser.add_argument('--device_id', type=int, default=0, help='GPU device ID')

    # evaluation parameters
    parser.add_argument('--eval_iter_nums', type=int, default=4, help='')

    return parser.parse_args()

###################Pre-Processing Tools########################
#
#
#
#
###############################################################


def get_principle_dirs(pts):

    pts_pca = PCA(n_components=3)
    pts_pca.fit(pts)
    principle_dirs = pts_pca.components_
    principle_dirs /= np.linalg.norm(principle_dirs, 2, axis=0)

    return principle_dirs

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)