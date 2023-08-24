from Model.Networks import ClassifierNet
from Dataset.DataLoader import PointcloudPatchDataset, RandomPointcloudPatchSampler, my_collate
from Dataset.RotatedViewGenerator import RotatedViewGenerator
from Dataset.RotatePatches import RotatePatches
from Utils import save_checkpoint
from Losses import regression_loss_fn

import os
from tqdm import tqdm
import numpy as np
import logging
import time

import torch
from torch.utils.tensorboard import SummaryWriter

class NormalEstimation:

    def __init__(self, opt, writer):
        self.opt = opt
        self.writer = writer
        self.checkpoint_path = opt.checkpoint_path
        self.shapes_list_file = opt.shapes_list_file
        self.patch_radius = opt.patch_radius
        self.points_per_patch = opt.points_per_patch
        self.cbs = opt.upstream_cbs
        self.alpha = opt.downstream_alpha
        self.beta = opt.downstream_beta
        self.delta = opt.downstream_delta
        self.power1 = 2
        self.power2 = opt.downstream_gamma
        self.device_id = opt.device_id

        self.model = ClassifierNet(3)

        if self.checkpoint_path is not None:
            checkpoint = torch.load(self.checkpoint_path, map_location='cuda')
            state_dict = checkpoint['state_dict']

            for k in list(state_dict.keys()):
                if k.startswith('embeddingnet_'):
                    del state_dict[k]

            self.model.load_state_dict(state_dict, strict=False)

            for name, param in self.model.named_parameters():
                if not name.startswith('classifier_'):
                    param.requires_grad = False

            print("Contrastive module loaded!")
            for name, param in self.model.named_parameters():
                if not name.startswith('classifier_'):
                    print(name, param.requires_grad)
        else:
            print("Contrastive module not loaded!")
            for name, param in self.model.named_parameters():
                if not name.startswith('classifier_'):
                    print(name, param.requires_grad)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.opt.lr, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

    def train(self):

        np.random.seed(self.opt.manualSeed)
        torch.manual_seed(self.opt.manualSeed)

        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)

        self.full_dataset = PointcloudPatchDataset(
            root=self.opt.trainset,
            shapes_list_file=self.shapes_list_file,
            patch_radius=self.patch_radius,
            points_per_patch=self.points_per_patch,
            seed=self.opt.manualSeed,
            train_state='train',
            train_type=self.opt.train_type,
            transform=RotatedViewGenerator(RotatePatches(2)),
            num_noise_levels=self.opt.num_noise_levels)

        self.train_datasampler = RandomPointcloudPatchSampler(
            self.full_dataset,
            patches_per_shape=self.opt.patches_per_shape,
            seed=self.opt.manualSeed,
            identical_epochs=False)

        self.train_dataloader = torch.utils.data.DataLoader(
            self.full_dataset,
            sampler=self.train_datasampler,
            shuffle=(self.train_datasampler is None),
            collate_fn=my_collate,
            batch_size=int(self.opt.batchSize),
            num_workers=int(self.opt.workers),
            pin_memory=True)

        if torch.cuda.is_available():
            with torch.cuda.device(self.device_id):
                self.model.to(device='cuda', dtype=torch.float)
            logging.info(f"Training with gpu: CUDA.")

        logging.info(f"Start classifier training for {self.opt.nepochs} epochs.")
        logging.info(f"Using checkpoint {self.checkpoint_path}.")
        logging.info(f"Regression batch size: {self.opt.batchSize}.")
        logging.info(f"Patches per shape: {self.opt.patches_per_shape}.")
        logging.info(f"Patch radius: {self.patch_radius}.")
        logging.info(f"Points per patch: {self.points_per_patch}.")
        logging.info(f"Number of noise levels: {self.opt.num_noise_levels}.")
        logging.info(f"Weight alpha: {self.alpha}.")
        logging.info(f"Weight beta: {self.beta}.")
        logging.info(f"Weight delta: {self.delta}.")
        logging.info(f"Weight gamma: {self.power2}.")

        training_losses = []
        epochs = []

        start_time = time.time()

        for epoch_counter in range(self.opt.nepochs):

            epoch_train_loss = 0.0
            running_train_loss = 0.0

            num_patches = 0

            print('\nTraining started for epoch {0}'.format(epoch_counter))

            for noisy_patches, gt_patches, gt_patch_normals, center_points, center_normals in tqdm(self.train_dataloader):
                num_patches += noisy_patches.size(0)

                if torch.cuda.is_available():
                    with torch.cuda.device(self.device_id):
                        noisy_patches = noisy_patches.to(device='cuda', dtype=torch.float)
                        gt_patches = gt_patches.to(device='cuda', dtype=torch.float)
                        gt_patch_normals = gt_patch_normals.to(device='cuda', dtype=torch.float)
                        center_points = center_points.to(device='cuda', dtype=torch.float)
                        center_normals = center_normals.to(device='cuda', dtype=torch.float)
                if noisy_patches.shape[0] <= 1:
                    continue

                preds = self.model(noisy_patches)
                pred_centers = preds[:,:3]
                pred_normals = preds[:,3:6]
                
                loss = regression_loss_fn(self.alpha, self.beta, self.delta, pred_centers, pred_normals, gt_patches, gt_patch_normals, self.power1, self.power2)

                # statistics
                running_train_loss += loss.item() * noisy_patches.size(0)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_train_loss = running_train_loss / num_patches
            self.scheduler.step(epoch_train_loss)

            logging.debug(f"Epoch: {epoch_counter}\tAccumulated training loss per batch: {epoch_train_loss}")
            logging.debug(f"                      \tLearning rate: {self.optimizer.param_groups[0]['lr']}")
            print(f"\nEpoch: {epoch_counter}\tAccumulated training loss per batch: {epoch_train_loss}")
            training_losses.append(epoch_train_loss)
            epochs.append(epoch_counter)

            if epoch_counter % 9 == 0:
                # save model checkpoints
                checkpoint_name = 'chkpt_cbs_{:02d}_ep{:02d}_a{:1.2f}_b{:1.2f}_d{:1.2f}_g{:02d}.pth.tar'.format(self.cbs, epoch_counter + 1, self.alpha, self.beta, self.delta, self.power2)
                checkpoint_path = os.path.join(self.writer.log_dir, checkpoint_name)
                save_checkpoint({
                    'epoch': epoch_counter + 1,
                    'arch': 'ClassifierNet',
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=False, filename=checkpoint_path)

        total_time = time.time() - start_time
        logging.info("Training has finished.")
        logging.debug(f"Total training time: {total_time}")