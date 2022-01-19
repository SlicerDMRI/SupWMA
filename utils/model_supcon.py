"""Reference from https://github.com/fxia22/pointnet.pytorch"""
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class PointNet_SupCon(nn.Module):
    """PointNet Encoder+Linear layers. Trained with contrastive loss"""
    def __init__(self, head='mlp', feat_dim=128, feature_transform=False, first_feature_transform=True, analy_critical_points=False,
                 use_recon_decoder=False):
        super(PointNet_SupCon, self).__init__()
        # encoder
        self.encoder = PointNetfeat(global_feat=True, feature_transform=feature_transform,
                                    first_feature_transform=first_feature_transform, analy_critical_points=analy_critical_points)
        # Contrastive learning
        if head == 'linear':
            self.head = nn.Linear(1024, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, feat_dim)
            )
        else:
            raise NotImplementedError(
                'Head not supported: {}. Please select from "mlp" or "linear"'.format(head))
        self.use_recon_decoder = use_recon_decoder
        if self.use_recon_decoder:
            # decoder
            self.decoder = PointNet_decoder()

    def forward(self, x):
        global_feat, _, _, point_idx = self.encoder(x)
        # contrastive feature
        contra_feat = F.normalize(self.head(global_feat), dim=1)
        if self.use_recon_decoder:
            # reconstruction loss
            recon_feat = F.normalize(self.decoder(global_feat), dim=2)
        else:
            recon_feat = torch.nn.Identity()

        return contra_feat, point_idx, recon_feat


class PointNet_Classifier(nn.Module):
    """The classifier layers in PointNet. Trained with CrossEntropy loss based on the fixed encoder"""
    def __init__(self, num_classes=2):
        super(PointNet_Classifier, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, first_feature_transform=True,
                 analy_critical_points=False, endpoints_contra=False):
        super(PointNetfeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        self.first_feature_transform = first_feature_transform
        self.analy_critical_points = analy_critical_points
        if self.first_feature_transform:
            self.stn = STN3d()
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        if self.first_feature_transform:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
        else:
            trans = None
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.analy_critical_points:
            _, point_idx = torch.max(x, 2)
        else:
            point_idx = None
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        return x, trans, trans_feat, point_idx


class PointNet_decoder(nn.Module):
    """A simple decoder for PointNet"""
    def __init__(self, num_points=15):
        super().__init__()
        self.num_points = num_points
        self.point_dim = 3
        self.fc1 = nn.Linear(in_features=1024, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=self.num_points*3)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        batch_size = x.shape[0]
        # decoder
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn3(self.fc2(x)))
        reconstructed_points = self.fc3(x)
        # do reshaping
        reconstructed_points = reconstructed_points.reshape(batch_size, self.point_dim, self.num_points)
        return reconstructed_points


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x