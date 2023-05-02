from __future__ import print_function

import torch
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate

import os
import copy
import numpy as np
import scipy.spatial as sp

from Utils import get_principle_dirs
from plyfile import PlyData


##################################New Dataloader Class###########################

def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))

    return default_collate(batch)


class RandomPointcloudPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source, patches_per_shape, seed=None, identical_epochs=False):
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.seed = seed
        self.identical_epochs = identical_epochs
        self.total_patch_count = None

        if self.seed is None:
            self.seed = np.random.random_integers(0, 2 ** 32 - 1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(self.data_source.shape_names):
            self.total_patch_count = self.total_patch_count + min(self.patches_per_shape,
                                                                  self.data_source.shape_patch_count[shape_ind])

    def __iter__(self):

        if self.identical_epochs:
            self.rng.seed(self.seed)

        return iter(
            self.rng.choice(sum(self.data_source.shape_patch_count), size=self.total_patch_count, replace=False))

    def __len__(self):
        return self.total_patch_count


class PointcloudPatchDataset(data.Dataset):

    def __init__(self, root=None, shapes_list_file=None, patch_radius=0.05, points_per_patch=500,
                 seed=None, train_state='train', shape_name=None, train_type=None, transform=None, num_noise_levels=0):

        self.root = root
        self.shapes_list_file = shapes_list_file

        self.patch_radius = patch_radius
        self.points_per_patch = points_per_patch
        self.seed = seed
        self.train_state = train_state
        self.train_type = train_type
        self.num_noise_levels = num_noise_levels

        # initialize rng for picking points in a patch
        if self.seed is None:
            self.seed = np.random.random_integers(0, 2 ** 10 - 1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.shape_patch_count = []
        self.patch_radius_absolute = []
        self.gt_shapes = []
        self.noise_shapes = []
        self.shape_names = []
        self.transform = transform

        if self.train_state == 'train':
            with open(os.path.join(self.root, self.shapes_list_file)) as f:
                all_shape_names = f.readlines()
            for shape_name in all_shape_names:
                if not shape_name.startswith('#'):
                    self.shape_names.append(shape_name.strip())

            self.shape_names = [x.strip() for x in self.shape_names]
            self.shape_names = list(filter(None, self.shape_names))
            for shape_ind, shape_name in enumerate(self.shape_names):
                print('getting information for shape %s' % shape_name)
                if shape_ind % self.num_noise_levels == 0:
                    gt_pts = np.load(os.path.join(self.root, shape_name + '.npy'))
                    gt_normal = np.load(os.path.join(self.root, shape_name + '_normal.npy'))
                    gt_kdtree = sp.cKDTree(gt_pts)
                    self.gt_shapes.append({'gt_pts': gt_pts, 'gt_normal': gt_normal, 'gt_kdtree': gt_kdtree})
                    self.noise_shapes.append({'noise_pts': gt_pts, 'noise_kdtree': gt_kdtree})
                    noise_pts = gt_pts
                else:
                    noise_pts = np.load(os.path.join(self.root, shape_name + '.npy'))
                    noise_kdtree = sp.cKDTree(noise_pts)
                    self.noise_shapes.append({'noise_pts': noise_pts, 'noise_kdtree': noise_kdtree})

                self.shape_patch_count.append(noise_pts.shape[0])
                bbdiag = float(np.linalg.norm(noise_pts.max(0) - noise_pts.min(0), 2))
                self.patch_radius_absolute.append(bbdiag * self.patch_radius)
        else:
            print('getting information for shape %s' % shape_name)
            noise_pts_all = PlyData.read(os.path.join(self.root, shape_name + '.ply'))['vertex'].data.tolist()
            noise_pts = np.asarray(noise_pts_all)[:, :3]
            noise_pts_normal = np.asarray(noise_pts_all)[:, 3:6]
            noise_kdtree = sp.cKDTree(noise_pts)
            self.noise_shapes.append({'noise_pts': noise_pts, 'noise_kdtree': noise_kdtree})

            self.shape_patch_count.append(noise_pts.shape[0])
            bbdiag = float(np.linalg.norm(noise_pts.max(0) - noise_pts.min(0), 2))
            self.patch_radius_absolute.append(bbdiag * self.patch_radius)

    def patch_sampling(self, patch_pts):

        if patch_pts.shape[0] > self.points_per_patch:

            sample_index = self.rng.choice(range(patch_pts.shape[0]), self.points_per_patch, replace=False)

        else:

            sample_index = self.rng.choice(range(patch_pts.shape[0]), self.points_per_patch)

        return sample_index

    def __getitem__(self, index):

        # find shape that contains the point with given global index

        shape_ind, patch_ind = self.shape_index(index)
        noise_shape = self.noise_shapes[shape_ind]
        noise_disp = copy.deepcopy(noise_shape['noise_pts'][patch_ind])
        patch_radius = copy.deepcopy(self.patch_radius_absolute[shape_ind])
        
        if self.train_state == 'train' and self.train_type == 'clearning':
            shape_ind_floor = shape_ind // self.num_noise_levels
            rand_num = int(np.floor(self.num_noise_levels * self.rng.rand()))
            contrastive_shape_index = self.num_noise_levels * shape_ind_floor + rand_num
            contrastive_noise_shape = self.noise_shapes[contrastive_shape_index]
            contrastive_patch_radius = copy.deepcopy(self.patch_radius_absolute[contrastive_shape_index])

        # For noise_patch
        noise_patch_idx = noise_shape['noise_kdtree'].query_ball_point(noise_disp, patch_radius)

        if len(noise_patch_idx) < 3:
            return None

        noise_patch_pts = copy.deepcopy(noise_shape['noise_pts'][noise_patch_idx]) - noise_disp
        noise_patch_pts /= patch_radius

        noise_sample_idx = self.patch_sampling(noise_patch_pts)
        noise_patch_pts = noise_patch_pts[noise_sample_idx]

        '''
        The transpose/inverse of the matrix Q comprising eigen-vectors in the 
        standard basis. Q would transform data from eigen-basis to 
        standard basis. We need inv(Q) which is the transpose
        '''
        noise_patch_eigenmatrix_inverse = get_principle_dirs(noise_patch_pts)

        if self.train_state == 'evaluation':
            noise_patch_pts = np.matmul(noise_patch_eigenmatrix_inverse, noise_patch_pts.T).T

            return torch.from_numpy(noise_patch_pts), \
                   torch.from_numpy(np.linalg.inv(noise_patch_eigenmatrix_inverse)), \
                   torch.from_numpy(np.array(patch_radius)), \
                   torch.from_numpy(noise_disp), \
                   torch.from_numpy(np.array(patch_ind))

        if self.train_state == 'train' and self.train_type == 'clearning':
            contrastive_noise_patch_idx = contrastive_noise_shape['noise_kdtree'].query_ball_point(
                noise_disp, contrastive_patch_radius)

            if len(contrastive_noise_patch_idx) < 3:
                return None

            contrastive_noise_patch_pts = copy.deepcopy(
                contrastive_noise_shape['noise_pts'][contrastive_noise_patch_idx]) - \
                                          noise_disp
            contrastive_noise_patch_pts /= contrastive_patch_radius

            contrastive_noise_sample_idx = self.patch_sampling(contrastive_noise_patch_pts)
            contrastive_noise_patch_pts = contrastive_noise_patch_pts[contrastive_noise_sample_idx]

        if self.train_type == 'regression':

            # For gt_patch
            gt_shape = self.gt_shapes[shape_ind // self.num_noise_levels]
            gt_patch_idx = gt_shape['gt_kdtree'].query_ball_point(noise_disp, patch_radius)

            if len(gt_patch_idx) < 3:
                return None

            gt_patch_pts = copy.deepcopy(gt_shape['gt_pts'][gt_patch_idx])
            gt_patch_pts -= noise_disp
            gt_patch_pts /= patch_radius

            gt_patch_normals = copy.deepcopy(gt_shape['gt_normal'][gt_patch_idx])

            gt_sample_idx = self.patch_sampling(gt_patch_pts)
            gt_patch_pts = gt_patch_pts[gt_sample_idx]
            gt_patch_normals = gt_patch_normals[gt_sample_idx]

            _, gt_nn_index = gt_shape['gt_kdtree'].query(noise_disp, 1)

            gt_center_point = copy.deepcopy(gt_shape['gt_pts'][gt_nn_index])
            gt_center_point -= noise_disp
            gt_center_point /= patch_radius

            gt_center_normal = copy.deepcopy(gt_shape['gt_normal'][gt_nn_index])

            noise_patch_pts = np.matmul(noise_patch_eigenmatrix_inverse, noise_patch_pts.T).T

            gt_patch_pts = np.matmul(noise_patch_eigenmatrix_inverse, gt_patch_pts.T).T
            gt_patch_normals = np.matmul(noise_patch_eigenmatrix_inverse, gt_patch_normals.T).T
            gt_center_point = np.squeeze(np.matmul(noise_patch_eigenmatrix_inverse, np.expand_dims(gt_center_point, axis=1)))
            gt_center_normal = np.squeeze(np.matmul(noise_patch_eigenmatrix_inverse, np.expand_dims(gt_center_normal, axis=1)))

            return torch.from_numpy(noise_patch_pts), \
                   torch.from_numpy(gt_patch_pts), \
                   torch.from_numpy(gt_patch_normals), \
                   torch.from_numpy(gt_center_point), \
                   torch.from_numpy(gt_center_normal)

        if self.train_type == 'clearning':
            noise_patch_pts, \
            contrastive_noise_patch_pts = self.transform('clearning',
                                                         self.rng,
                                                         [noise_patch_pts,
                                                          None,
                                                          None,
                                                          contrastive_noise_patch_pts,
                                                          None])

        noise_patch_pts = self.put_vector_bundle_into_eigenbasis(noise_patch_eigenmatrix_inverse, noise_patch_pts)
        contrastive_noise_patch_pts = self.put_vector_bundle_into_eigenbasis(noise_patch_eigenmatrix_inverse, contrastive_noise_patch_pts)

        all_transformed_patches = [noise_patch_pts[1], contrastive_noise_patch_pts[0], contrastive_noise_patch_pts[1]]
        transformed_patch_pick = int(np.floor(3 * self.rng.rand()))

        if self.train_type == 'clearning':
            return [torch.from_numpy(noise_patch_pts[0]),
                    torch.from_numpy(all_transformed_patches[transformed_patch_pick])]

    def put_vector_bundle_into_eigenbasis(self, noise_patch_eigenmatrix_inverse, xs):
        ys = []
        for x in xs:
            ys.append(np.matmul(noise_patch_eigenmatrix_inverse, x.T).T)

        return ys

    def put_vector_into_eigenbasis(self, noise_patch_eigenmatrix_inverse, xs):
        ys = []
        for x in xs:
            ys.append(np.squeeze(np.matmul(noise_patch_eigenmatrix_inverse, np.expand_dims(x, axis=1))))

        return ys

    def __len__(self):
        return sum(self.shape_patch_count)

    def shape_index(self, index):
        shape_patch_offset = 0
        shape_ind = None
        for shape_ind, shape_patch_count in enumerate(self.shape_patch_count):
            if (index >= shape_patch_offset) and (index < shape_patch_offset + shape_patch_count):
                shape_patch_ind = index - shape_patch_offset
                break
            shape_patch_offset = shape_patch_offset + shape_patch_count

        return shape_ind, shape_patch_ind
