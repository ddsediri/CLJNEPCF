from Model.Networks import EmbeddingNet
from Dataset.DataLoader import PointcloudPatchDataset, RandomPointcloudPatchSampler, my_collate
from Dataset.RotatedViewGenerator import RotatedViewGenerator
from Dataset.RotatePatches import RotatePatches
from Utils import save_checkpoint

import os
from tqdm import tqdm
import numpy as np
import logging
import time

import torch
import torch.nn.functional as F

class ContrastiveLearning:

    def __init__(self, opt, writer):
        self.opt = opt
        self.writer = writer
        self.shapes_list_file = opt.shapes_list_file
        self.patch_radius = opt.patch_radius
        self.points_per_patch = opt.points_per_patch
        self.device_id = opt.device_id
        self.model = EmbeddingNet(3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.opt.lr, weight_decay=self.opt.wd)

        with torch.cuda.device(self.device_id):
            self.criterion = torch.nn.CrossEntropyLoss().to('cuda')

    def contrastive_learning_loss_fn(self, features):
        n_views = 2

        labels = torch.cat([torch.arange(len(features) / n_views) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        with torch.cuda.device(self.device_id):
            labels = labels.to('cuda')

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        del features

        # discard the main diagonal from both: labels and similarities matrix
        with torch.cuda.device(self.device_id):
            mask = torch.eye(labels.shape[0], dtype=torch.bool).to('cuda')

        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select and combine multiple negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        del positives, negatives, similarity_matrix

        with torch.cuda.device(self.device_id):        
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to('cuda')

        logits = logits / 0.01 #0.07
        return logits, labels

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
            batch_size=self.opt.batchSize,
            num_workers=int(self.opt.workers),
            pin_memory=True)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=2*self.opt.nepochs,
                                                                    eta_min=0,
                                                                    last_epoch=-1)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)

        if torch.cuda.is_available():
            with torch.cuda.device(self.device_id):
                self.model.to(device='cuda', dtype=torch.float)
            logging.info(f"Training with gpu: CUDA.")

        logging.info(f"Start feature encoder training for {self.opt.nepochs} epochs.")
        logging.info(f"Contrastive learning batch size: {self.opt.batchSize}.")
        logging.info(f"Patches per shape: {self.opt.patches_per_shape}.")
        logging.info(f"Patch radius: {self.patch_radius}.")
        logging.info(f"Points per patch: {self.points_per_patch}.")
        logging.info(f"Number of noise levels: {self.opt.num_noise_levels}.")

        n_iter = 0

        losses = []
        epochs = []

        start_time = time.time()

        for epoch_counter in range(self.opt.nepochs):

            epoch_loss = 0.0
            running_loss = 0.0
            num_patches = 0

            print('\nTraining started for epoch {0}'.format(epoch_counter))

            for noisy_patches in tqdm(self.train_dataloader):
                noisy_patches = torch.cat(noisy_patches, dim=0)
                batch_size = int(noisy_patches.size(0))
                num_patches += batch_size

                if torch.cuda.is_available():
                    with torch.cuda.device(self.device_id):
                        noisy_patches = noisy_patches.float().cuda(non_blocking=True)
                if batch_size <= 1:
                    continue

                features = self.model(noisy_patches)
                del noisy_patches
                logits, labels = self.contrastive_learning_loss_fn(features)
                del features
                loss = self.criterion(logits, labels)
                del logits, labels

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # statistics
                running_loss += float(loss) * batch_size 

            # warmup for the first 5 epochs
            if epoch_counter >= 5:
                self.scheduler.step()

            epoch_loss = float(running_loss / num_patches)
            print(epoch_loss)

            logging.debug(f"Epoch: {epoch_counter}\tAccumulated loss per batch: {epoch_loss}")
            logging.debug(f"                      \tLearning rate: {self.scheduler.get_last_lr()[0]}")
            print(f"\nEpoch: {epoch_counter}\tAccumulated loss per batch: {epoch_loss}")
            losses.append(epoch_loss)
            epochs.append(epoch_counter)

            if epoch_counter % 9 == 0:
                # save model checkpoints
                checkpoint_name = 'chkpt_cbs_{:02d}_ep{:02d}.pth.tar'.format(self.opt.batchSize, epoch_counter + 1)
                checkpoint_path = os.path.join(self.writer.log_dir, checkpoint_name)
                save_checkpoint({
                    'epoch': epoch_counter + 1,
                    'arch': 'EmbeddingNet',
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=False, filename=checkpoint_path)

        total_time = time.time() - start_time
        logging.info("Training has finished.")
        logging.debug(f"Total training time: {total_time}")

        return checkpoint_path