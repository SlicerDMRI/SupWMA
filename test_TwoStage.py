from utils.logger import create_logger

import whitematteranalysis as wma
import numpy as np

import argparse
import h5py
import time
import os
import pickle

import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data

from utils.model import PointNetCls
from utils.model_supcon import PointNet_SupCon, PointNet_Classifier
from utils.dataset import TestDataset
from utils.metrics_plots import classify_report


def load_test_data():
    """Load test data and labels name in model"""
    # Put test data into loader
    test_dataset = TestDataset(args.feat_path, args.input_label_path, args.label_names,
                               script_name)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size,
        shuffle=False, num_workers=int(args.num_workers))
    test_data_size = len(test_dataset)
    print(script_name, 'The test data size is:{}'.format(test_data_size))
    num_classes = len(test_dataset.label_names_in_model)
    # load label names
    label_names = test_dataset.label_names_in_model
    print('The label names are: {}'.format(str(label_names)))
    print(script_name, 'The number of classes is:{}'.format(num_classes))

    return test_loader, label_names, num_classes


def load_model():
    """load stage 1 and stage 2 models"""
    stage1_model = PointNetCls(k=stage1_params['stage1_num_class']).to(device)

    stage2_encoder = PointNet_SupCon(head=encoder_params['head_name'], feat_dim=encoder_params['encoder_feat_num']).to(device)

    stage2_classifer = PointNet_Classifier(num_classes=encoder_params['stage2_num_class']).to(device)

    # load weights
    stage1_weight_path = os.path.join(args.weight_path, 's1_cls', 'best_{}_model.pth'.format(args.best_metric))
    stage1_model.load_state_dict(torch.load(stage1_weight_path))
    encoder_weight_path = os.path.join(args.weight_path, 's2_encoder', 'epoch_{}_model.pth'.format(args.supcon_epoch))
    stage2_encoder.load_state_dict(torch.load(encoder_weight_path))
    classifier_weight_path = os.path.join(args.weight_path, 's2_cls', 'best_{}_model.pth'.format(args.best_metric))
    stage2_classifer.load_state_dict(torch.load(classifier_weight_path))
    return stage1_model, stage2_encoder, stage2_classifer


def test_net():
    """perform predition of multiple clusters"""
    print('')
    print('===================================')
    print('')
    print(script_name, 'Start multi-cluster prediction.')

    output_prediction_mask_path = os.path.join(args.out_path, args.out_prefix + '_test_prediction_mask.h5')
    output_prediction_report_path = os.path.join(args.out_path, args.out_prefix + '_test_prediction_report.h5')
    stage1_net, stage2_encoder_net, stage2_classifer_net = load_model()
    if not os.path.exists(output_prediction_mask_path):
        # Load model
        start_time = time.time()
        with torch.no_grad():
            total_test_correct = 0
            test_labels_lst, test_predicted_lst = [], []
            for j, data in (enumerate(test_data_loader, 0)):
                points, labels = data
                points = points.transpose(2, 1)
                if args.input_label_path is not None:
                    labels = labels[:, 0]
                    points, labels = points.to(device), labels.to(device)
                else:
                    points = points.to(device)
                stage1_net, stage2_encoder_net, stage2_classifer_net = \
                    stage1_net.eval(), stage2_encoder_net.eval(), stage2_classifer_net.eval()

                # initialization
                tmp = torch.tensor(-1).to(device)
                pred_idx = tmp.repeat(points.shape[0])
                # stage 1
                stage1_pred = stage1_net(points)
                _, stage1_pred_idx = torch.max(stage1_pred, dim=1)
                stage1_swm_mask = torch.where(stage1_pred_idx < stage1_params['num_swm_stage1'])[0]
                stage1_other_mask = torch.where(stage1_pred_idx >= stage1_params['num_swm_stage1'])[0]
                pred_idx[stage1_other_mask] = torch.tensor(198).to(device)

                # stage 2
                if stage1_swm_mask.shape[0] != 0:
                    swm_points = points[stage1_swm_mask, :, :]
                    features = stage2_encoder_net.encoder(swm_points)
                    stage2_pred = stage2_classifer_net(features)
                    _, stage2_pred_idx = torch.max(stage2_pred, dim=1)
                    pred_idx[stage1_swm_mask] = torch.where(stage2_pred_idx < 198, stage2_pred_idx, torch.tensor(198).to(device))
                
                # entire data
                if args.input_label_path is not None:
                    # for classification report
                    correct = pred_idx.eq(labels.data).cpu().sum()
                    # for calculating test accuracy
                    total_test_correct += correct.item()
                    # for calculating test weighted and macro metrics
                    labels = labels.cpu().detach().numpy().tolist()
                    test_labels_lst.extend(labels)
                assert torch.sum(pred_idx == tmp) == 0
                pred_idx = pred_idx.cpu().detach().numpy().tolist()
                test_predicted_lst.extend(pred_idx)
        end_time = time.time()
        print('The total time of prediction is:{} s'.format(round((end_time - start_time), 4)))
        print('The test sample size is: ', len(test_predicted_lst))
        if args.input_label_path is not None:
            classify_report(test_labels_lst, test_predicted_lst, label_names, logger, output_prediction_report_path, 'test')

        test_prediction_lst_h5 = h5py.File(output_prediction_mask_path, "w")
        test_prediction_lst_h5['complete_pred_test'] = test_predicted_lst
        test_predicted_array = np.asarray(test_predicted_lst)

    else:
        print(script_name, 'Loading prediction result.')
        test_prediction_h5 = h5py.File(output_prediction_mask_path, "r")
        test_predicted_array = np.asarray(test_prediction_h5['complete_pred_test'])

    return test_predicted_array


def tractography_parcellation():
    """Generate the tractography parcellation results with the predicted list"""
    output_cluster_folder = None
    if args.tractography_path is not None:
        print('')
        print('===================================')
        print('')
        print(script_name, 'Output fiber clusters.')
        # Tractography Parcellation
        cluster_prediction_mask = predicted_arr
        print(script_name, 'Load vtk:', args.tractography_path)
        pd_whole_cluster = wma.io.read_polydata(args.tractography_path)
        number_of_clusters = np.max(cluster_prediction_mask) + 1
        pd_t_list = wma.cluster.mask_all_clusters(pd_whole_cluster, cluster_prediction_mask, number_of_clusters,
                                                  preserve_point_data=False, preserve_cell_data=False, verbose=False)
        output_cluster_folder = os.path.join(args.out_path, args.out_prefix + '_prediction_clusters_outlier_removed')

        if not os.path.exists(output_cluster_folder):
            os.makedirs(output_cluster_folder)

        for t_idx in range(len(pd_t_list)):
            pd_t = pd_t_list[t_idx]

            if label_names is not None:
                fname_t = os.path.join(output_cluster_folder, label_names[t_idx] + '.vtp')
            else:
                fname_t = os.path.join(output_cluster_folder, 'cluster_' + str(t_idx) + '.vtp')

            print(script_name, 'output', fname_t)
            wma.io.write_polydata(pd_t, fname_t)

        print(script_name, 'Done! Clusters are in:', output_cluster_folder)

    return output_cluster_folder


if __name__ == "__main__":
    use_cpu = False
    if use_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")

    # Parse arguments
    parser = argparse.ArgumentParser(description="Testing on real data using Two-stage SupWMA",
                                     epilog="by Tengfei Xue txue4133@uni.sydney.edu.au")
    parser.add_argument('--weight_path', type=str, help='pretrained network model')
    parser.add_argument('--feat_path', type=str, help='Input cluster feature data as an h5 file.')
    parser.add_argument('--out_path', type=str, help='The output directory should be a new empty directory. It will be created if needed.')
    parser.add_argument('--label_names', type=str, help='label names in the trained model as an h5 file.')
    parser.add_argument('--input_label_path', type=str, help='Input ground truth label as an h5 file.')
    parser.add_argument('--out_prefix', type=str, help='A prefix string of all output files.')
    parser.add_argument('--tractography_path', type=str, help='Tractography data as a vtkPolyData file. If given, prediction will output clusters')

    parser.add_argument('--num_workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--test_batch_size', type=int, default=6144, help='batch size')

    parser.add_argument('--best_metric', type=str, default='f1', help='evaluation metric')
    parser.add_argument('--supcon_epoch', type=int, default=100, help='The epoch of encoder model')

    args = parser.parse_args()
    script_name = '<test-TwoStage>'

    logger = create_logger(args.out_path)

    if not os.path.exists(args.out_path):
        print(script_name, "Output directory", args.out_path, "does not exist, creating it.")
        os.makedirs(args.out_path)

    with open(os.path.join(args.weight_path, 's1_cls', 'stage1_params.pickle'), 'rb') as f:
        stage1_params = pickle.load(f)
        f.close()
    with open(os.path.join(args.weight_path, 's2_encoder', 'encoder_params.pickle'), 'rb') as f:
        encoder_params = pickle.load(f)
        f.close()

    # load test data
    test_data_loader, label_names, num_class = load_test_data()

    # generate prediction
    predicted_arr = test_net()

    # Process tractography parcellation
    out_clusters_folder = tractography_parcellation()
