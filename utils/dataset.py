from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
from tqdm import tqdm
import json
from plyfile import PlyData, PlyElement
import h5py
import sys
sys.path.append('../')
import utils.tract_feat as tract_feat


class TestDataset(data.Dataset):
    def __init__(self, test_features_path, test_label_path,
                 label_names_path, script_name, normalization=False,
                 data_augmentation=False):
        self.feat_p = test_features_path
        self.label_p = test_label_path
        self.label_names_p = label_names_path
        self.script_name = script_name
        self.data_augmentation = data_augmentation
        self.normalization = normalization
        # load label names
        print(self.script_name, 'Load clusters names along with the model.')
        labels_names_in_model_h5 = h5py.File(self.label_names_p, 'r')
        self.label_names_in_model = labels_names_in_model_h5['y_names']
        self.label_names_in_model = [name.decode('UTF-8') for name in self.label_names_in_model]
        # load feature data
        print(self.script_name, 'Load input feature.')
        feat_h5 = h5py.File(self.feat_p, 'r')
        # self.features = np.asarray(feat_h5['feat']).squeeze()
        self.features = np.asarray(feat_h5['feat']).squeeze(-1)
        # load label data
        if self.label_p is not None:
            print(self.script_name, 'Load input label.')
            label_h5 = h5py.File(self.label_p, 'r')
            label_test_gt = label_h5['label_array'].value.astype(int)
            label_names = label_h5['label_names'].value
            # Generate final ground truth label
            print(self.script_name, 'Generate FINAL ground truth label for evaluation.')
            self.label_test_final = tract_feat.update_y_test_based_on_model_y_names(label_test_gt, label_names,
                                                                                    self.label_names_in_model)
        else:
            self.label_test_final = None
        print('The size of feature for is {}'.format(self.features.shape))

    def __getitem__(self, index):
        point_set = self.features[index]
        if self.label_p is not None:
            label = self.label_test_final[index]
        else:
            label = None
        if self.normalization:
            point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
            dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
            point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        if point_set.dtype == 'float32':
            point_set = torch.from_numpy(point_set)
        else:
            point_set = torch.from_numpy(point_set.astype(np.float32))
            # print('Feature is not in float32 format')

        if label is None:
            # None can not be passed
            label = torch.from_numpy(np.asarray([1]))
        elif label.dtype == 'int64':
            label = torch.from_numpy(np.array([label]))
        else:
            label = torch.from_numpy(np.array([label]).astype(np.int64))
            # print('Label is not in int64 format')

        return point_set, label

    def __len__(self):
        return self.features.shape[0]
