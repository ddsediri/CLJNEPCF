import torch
import torch.nn as nn
import torch.nn.functional as F


class BasePointNet(nn.Module):

    def __init__(self, point_dimension):
        super(BasePointNet, self).__init__()

        self.conv_1 = nn.Conv1d(point_dimension, 64, 1)
        self.conv_2 = nn.Conv1d(64, 128, 1)
        self.conv_3 = nn.Conv1d(128, 256, 1)
        self.conv_4 = nn.Conv1d(256, 512, 1)
        self.conv_5 = nn.Conv1d(512, 1024, 1)

        self.bn_1 = nn.BatchNorm1d(64)
        self.bn_2 = nn.BatchNorm1d(128)
        self.bn_3 = nn.BatchNorm1d(256)
        self.bn_4 = nn.BatchNorm1d(512)
        self.bn_5 = nn.BatchNorm1d(1024)

    def forward(self, x):
        num_points = x.shape[1]
        x = x.transpose(2, 1)

        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))
        x = F.relu(self.bn_4(self.conv_4(x)))
        x = F.relu(self.bn_5(self.conv_5(x)))
        x = nn.MaxPool1d(num_points)(x)
        x = x.view(-1, 1024)

        return x

class EmbeddingNet(nn.Module):

    def __init__(self, point_dimension, return_local_features=False):
        super(EmbeddingNet, self).__init__()
        self.base_pointnet = BasePointNet(point_dimension)

        self.embeddingnet_fc_1 = nn.Linear(1024, 1024)
        self.embeddingnet_fc_2 = nn.Linear(1024, 512)
        self.embeddingnet_fc_3 = nn.Linear(512, 256)

        self.embeddingnet_bn_1 = nn.BatchNorm1d(1024)
        self.embeddingnet_bn_2 = nn.BatchNorm1d(512)

    def forward(self, x):
        x = self.base_pointnet(x)

        x = F.relu(self.embeddingnet_bn_1(self.embeddingnet_fc_1(x)))
        x = F.relu(self.embeddingnet_bn_2(self.embeddingnet_fc_2(x)))
        x = self.embeddingnet_fc_3(x)

        return x

class ClassifierNet(nn.Module):

    def __init__(self, point_dimension, return_local_features=False):
        super(ClassifierNet, self).__init__()
        self.base_pointnet = BasePointNet(point_dimension)

        self.classifier_fc_1 = nn.Linear(1024, 512)
        self.classifier_fc_2 = nn.Linear(512, 256)
        self.classifier_fc_3 = nn.Linear(256, 128)
        self.classifier_fc_4 = nn.Linear(128, 64)
        self.classifier_fc_5 = nn.Linear(64, 6)

        self.classifier_bn_1 = nn.BatchNorm1d(512)
        self.classifier_bn_2 = nn.BatchNorm1d(256)
        self.classifier_bn_3 = nn.BatchNorm1d(128)
        self.classifier_bn_4 = nn.BatchNorm1d(64)

        self.dropout_1 = nn.Dropout(0.3)
        self.dropout_2 = nn.Dropout(0.3)
        self.dropout_3 = nn.Dropout(0.3)
        self.dropout_4 = nn.Dropout(0.3)

    def forward(self, x):
        x = self.base_pointnet(x)

        x = F.relu(self.classifier_bn_1(self.classifier_fc_1(x)))
        x = self.dropout_1(x)
        x = F.relu(self.classifier_bn_2(self.classifier_fc_2(x)))
        x = self.dropout_2(x)
        x = F.relu(self.classifier_bn_3(self.classifier_fc_3(x)))
        x = self.dropout_3(x)
        x = F.relu(self.classifier_bn_4(self.classifier_fc_4(x)))
        x = self.dropout_4(x)
        x = torch.tanh(self.classifier_fc_5(x))

        center = x[:, :3]
        normal = x[:, 3:6]

        norm = torch.norm(normal, dim=1).view(-1, 1)
        normal = normal / norm

        x = torch.cat([center, normal], dim=1)

        return x