# coding=utf-8

from RegressorTrain import NormalEstimation
from Utils import parse_arguments
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


if __name__ == '__main__':

    opt = parse_arguments()

    log_dir = './RegressorTrained'

    writer = SummaryWriter(log_dir=log_dir)

    classifier = NormalEstimation(opt,
        writer)
    
    classifier.train()
