import numpy as np
from math import sin, cos


class RotatePatches(object):
    """Rotate patches, this is our transformation for the contrastive dataset creation"""

    def __init__(self, n_views=2):
        self.n_views = n_views

    def __call__(self, rng, all_old_patches_and_vecs):

        all_new_patches_and_vecs = []
        select_axis = rng.randint(3)
        thetas = [np.pi/12, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 7*np.pi/12, 2*np.pi/3, 3*np.pi/4, 5*np.pi/6, np.pi]
        select_theta = rng.randint(10)

        for old_patch_or_vec in all_old_patches_and_vecs:
            new_patches_or_vecs = []

            for i in range(self.n_views):
                if i == 0:
                    m = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                else:
                    theta = thetas[select_theta] #np.pi

                    if select_axis == 0:
                        # rotate the samples by theta radians around x
                        m = [[1, 0, 0], [0, cos(theta), -sin(theta)], [0, sin(theta), cos(theta)]]
                    elif select_axis == 1:
                        # rotate the samples by theta radians around y
                        m = [[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]]
                    elif select_axis == 2:
                        # rotate the samples by theta radians around z
                        m = [[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]]

                if old_patch_or_vec is not None and old_patch_or_vec.shape[0] > 3:
                    new_noisy_patch_or_vec = []

                    for noisy_vec in old_patch_or_vec:
                        new_noisy_patch_or_vec.append(np.dot(m, noisy_vec))

                    new_patches_or_vecs.append(np.array(new_noisy_patch_or_vec))
                elif old_patch_or_vec is not None and old_patch_or_vec.shape[0] == 3:
                    new_patches_or_vecs.append(np.array(np.dot(m, old_patch_or_vec)))
                elif old_patch_or_vec is None:
                    new_patches_or_vecs.append([])

            all_new_patches_and_vecs.append(new_patches_or_vecs)

        return all_new_patches_and_vecs
