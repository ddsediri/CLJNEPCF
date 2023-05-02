import numpy as np
from torchvision.transforms import transforms
from Dataset.RotatePatches import RotatePatches


class RotatedViewGenerator(object):
    """Create two views of the same patch and generate the dataset of contrasted pairs."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self,
                 train_type,
                 rng,
                 old_patches_and_vecs):

        new_noisy_patches, \
        new_center_points, \
        new_center_normals, \
        new_contrastive_noise_patch, \
        new_cn_gt_center_normal = self.base_transform(rng, old_patches_and_vecs)

        if train_type == 'clearning':
            return [new_noisy_patches[0], new_noisy_patches[1]], \
                   [new_contrastive_noise_patch[0], new_contrastive_noise_patch[1]], \

        else:
            return [new_noisy_patches[0], new_noisy_patches[1]], \
                   [new_center_points[0], new_center_points[1]], \
                   [new_center_normals[0], new_center_normals[1]], \
                   [new_contrastive_noise_patch[0], new_contrastive_noise_patch[1]], \
                   [new_cn_gt_center_normal[0], new_cn_gt_center_normal[1]]
