# coding=utf-8

from Dataset.DataLoader import PointcloudPatchDataset, my_collate
from Model.Networks import ClassifierNet
from Utils import parse_arguments

import os
import numpy as np
import scipy.spatial as sp
from tqdm import tqdm
from plyfile import PlyData, PlyElement

import torch
from sklearn.neighbors import NearestNeighbors
import copy
import time

def npy2ply(pts, save_filename):
    vertex = [tuple(item) for item in pts]
    vertex = np.array(vertex, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                     ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
    PlyData([PlyElement.describe(vertex, 'vertex')], text=False).write(save_filename)

def check_noise_level(shape_name):
    return '_' + shape_name.split('_')[-1]

def get_meaned_displacements(shp, moved_points, n_neighbours):
    shp_kdtree = sp.cKDTree(shp)
    nearest_neighbours = torch.tensor(shp_kdtree.query(shp, n_neighbours)[1])
    displacement_vectors = moved_points - shp
    new_displacement = displacement_vectors[nearest_neighbours]
    new_displacement = new_displacement.mean(1)
    new_points = moved_points - new_displacement
    return new_points

def get_new_centers(noise_pts, pred_normals, n_neighbours):
    new_point_list = copy.deepcopy(noise_pts)
    noise_pts_kdtree = sp.cKDTree(noise_pts)
    num_rng0 = noise_pts_kdtree.query(noise_pts, n_neighbours)

    num_rng_new = copy.deepcopy(num_rng0)
    num_rng = (num_rng_new[0][:,1:], num_rng_new[1][:,1:])

    point_offset = new_point_list[num_rng[1]] - np.tile(np.expand_dims(new_point_list, axis=1), (1, n_neighbours-1, 1))
    m1 = np.expand_dims(pred_normals[num_rng[1]], axis=2)
    m1Tm1 = np.matmul(m1.swapaxes(-1,-2), m1)
    m2 = np.expand_dims(pred_normals, axis=1)
    m2Tm2 = np.matmul(m2.swapaxes(-1,-2), m2)
    m2Tm2 = np.tile(np.expand_dims(m2Tm2, axis=1), (1, n_neighbours-1, 1, 1))
    m = m1Tm1 + m2Tm2
    gamma = 1 / (3 * (n_neighbours-1))
    point_offset = np.expand_dims(point_offset, axis=2)
    v1 = gamma*(np.matmul(point_offset, m).sum(1))
    new_point_list_2 = new_point_list + np.squeeze(v1, axis=1)

    return new_point_list_2

if __name__ == '__main__':

    opt = parse_arguments()
    opt.num_noise_levels = 0

    shapes_list_file = 'testset.txt'
    dataset_root = './Dataset/'
    testset_root = dataset_root + 'TestHighNoise'
    test_results_type = 'TestHighNoiseResults'
    opt.save_dir = dataset_root + test_results_type
    gt_root = './Dataset/GroundTruth'

    model = ClassifierNet(3)

    print("Patch radius during testing is set to: {}".format(opt.patch_radius))
    print("Points per patch during testing is set to: {}".format(opt.points_per_patch))

    save_corrected_gt_points = True # Save corrected GTs to account for outlier removal
    neighbourhood_size = 20 # Neighbourhood size for LRMA update

    default_num_tot_iter = opt.eval_iter_nums # For noise scales < 2% of the bounding box diagonal, we use 4 denoising iterations
    default_hn_num_tot_iter = 10 # For noise scales > 2% of the bounding box diagonal, we use 10 denoising iterations
    
    start_time = time.time()

    results_dir = opt.save_dir

    print("Results will be saved at: {}".format(results_dir))
    print("Checkpoint path is located at: {}".format(opt.checkpoint_path))

    try:
        os.makedirs(results_dir)
    except FileExistsError:
        # directory already exists
        pass

    checkpoint = torch.load(opt.checkpoint_path, map_location='cuda')
    state_dict = checkpoint['state_dict']

    model.load_state_dict(state_dict, strict=False)
    if torch.cuda.is_available():
        model.to(device='cuda', dtype=torch.float)

    model.eval()

    shape_names = []
    with open(os.path.join(testset_root, shapes_list_file)) as f:
        all_shape_names = f.readlines()
    for shape_name in all_shape_names:
        if not shape_name.startswith('#'):
            shape_names.append(shape_name)

    shape_names = [x.strip() for x in shape_names]
    shape_names = list(filter(None, shape_names))

    for shape_name in shape_names:

        if shape_name.endswith('_0.02') or shape_name.endswith('_0.020') or shape_name.endswith('_0.025'):
            num_tot_iter = default_hn_num_tot_iter
        else:
            num_tot_iter = default_num_tot_iter

        print("Number of filtering iterations: {}".format(default_num_tot_iter))

        gt_noise_level = check_noise_level(shape_name)
        gt_root_save_dir = dataset_root + 'Corrected_GT_' + test_results_type

        if save_corrected_gt_points:
            try:
                os.makedirs(gt_root_save_dir)
            except FileExistsError:
                # directory already exists
                pass
        
        for iter in range(num_tot_iter):
            shape_name_iter = '{}_{}'.format(shape_name, iter)

            if iter == 0:
                noise_pts_all = np.asarray(PlyData.read(os.path.join(testset_root, shape_name + '.ply'))['vertex'].data.tolist())
                assert noise_pts_all.ndim == 2, "Please make sure the point cloud has dimensions (N, D)."
                assert noise_pts_all.shape[-1] >= 6, "Please make sure points have at least 6 dimensions. 3 for position and 3 for the associated PCA normal."
                noise_pts = noise_pts_all[:, :3]
                pca_normals = noise_pts_all[:, 3:6]
                save_pts = np.append(noise_pts, pca_normals, axis=1)
                npy2ply(save_pts, os.path.join(results_dir, shape_name_iter + '.ply'))

                if save_corrected_gt_points:
                    gt_pts_all = np.asarray(PlyData.read(os.path.join(gt_root, shape_name.replace(gt_noise_level, '') + '.ply'))['vertex'].data.tolist())
                    gt_pts = gt_pts_all[:, :3]
                    gt_normals = gt_pts_all[:, 3:6]
                    save_gt_pts = np.append(gt_pts, gt_normals, axis=1)
                    npy2ply(save_gt_pts, os.path.join(gt_root_save_dir, shape_name + '.ply'))
            else:
                noise_pts_all = np.asarray(PlyData.read(os.path.join(results_dir, shape_name_iter + '.ply'))['vertex'].data.tolist())
                noise_pts = noise_pts_all[:, :3]
                pca_normals = noise_pts_all[:, 3:6]

                if save_corrected_gt_points:
                    gt_pts_all = np.asarray(PlyData.read(os.path.join(gt_root_save_dir, shape_name + '.ply'))['vertex'].data.tolist())
                    gt_pts = gt_pts_all[:, :3]
                    gt_normals = gt_pts_all[:, 3:6]

            pred_centers = np.empty((0, 3), dtype='float32')
            pred_normals = np.empty((0, 3), dtype='float32')
            patch_indices = np.empty((0), dtype='int')

            test_dataset = PointcloudPatchDataset(
                root=results_dir,
                patch_radius=opt.patch_radius,
                points_per_patch=opt.points_per_patch,
                seed=opt.manualSeed,
                train_state='evaluation',
                shape_name=shape_name_iter,
                transform=None,
                num_noise_levels=opt.num_noise_levels)

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                sampler=None,
                shuffle=None,
                collate_fn=my_collate,
                batch_size=50,
                num_workers=int(opt.workers))

            # if iter < num_tot_iter - 1:
            #     os.remove(os.path.join(results_dir, shape_name_iter + '.ply'))
            shape_name_iter = '{}_{}'.format(shape_name, iter + 1)
            patch_radius = test_dataset.patch_radius_absolute

            for noise_patch, transformation_to_standard_basis, _, noise_disp, patch_ind in tqdm(test_dataloader):
                if torch.cuda.is_available():
                    noise_patch = noise_patch.to(device='cuda', dtype=torch.float)
                    transformation_to_standard_basis = transformation_to_standard_basis.to(device='cuda', dtype=torch.float)

                pred = model(noise_patch)
                pred_center = pred[:,:3]
                pred_normal = pred[:,3:6]

                pred_center = torch.bmm(transformation_to_standard_basis, pred_center.unsqueeze(2))
                pred_normal = torch.bmm(transformation_to_standard_basis, pred_normal.unsqueeze(2))

                pred_centers = np.append(pred_centers,
                                         np.squeeze(pred_center.data.cpu().numpy()) * patch_radius + noise_disp.numpy(),
                                         axis=0)
                pred_normals = np.append(pred_normals,
                                         np.squeeze(pred_normal.data.cpu().numpy()),
                                         axis=0)
                patch_indices = np.append(patch_indices,
                                          patch_ind.numpy(),
                                          axis=0)

            i = 0
            for index in tqdm(patch_indices):
                pred_check = np.matmul(np.expand_dims(pca_normals[index],axis=0), np.expand_dims(pred_normals[i],axis=1))
                if pred_check < 0:
                    pred_normals[i] = -pred_normals[i]
                i += 1

            idx_to_del = []

            for index, _ in enumerate(tqdm(pca_normals)):
                if index not in patch_indices:
                    idx_to_del.append(index)

            noise_pts = np.delete(noise_pts, idx_to_del, axis=0)
            pca_normals = np.delete(pca_normals, idx_to_del, axis=0)

            if save_corrected_gt_points:
                gt_pts = np.delete(gt_pts, idx_to_del, axis=0)
                gt_normals = np.delete(gt_normals, idx_to_del, axis=0)

            pred_centers = get_meaned_displacements(noise_pts, pred_centers, 100)

            pred_2_centers = get_new_centers(pred_centers, pred_normals, neighbourhood_size)

            save_pts = np.append(pred_2_centers, pred_normals, axis=1)
            npy2ply(save_pts, os.path.join(results_dir, shape_name_iter + '.ply'))

            if save_corrected_gt_points:
                save_gt_pts = np.append(gt_pts, gt_normals, axis=1)
                npy2ply(save_gt_pts, os.path.join(gt_root_save_dir, shape_name + '.ply'))

    end_time = time.time()

    tot_time = end_time - start_time
    print("Total time taken on test times set: {}s".format(tot_time))