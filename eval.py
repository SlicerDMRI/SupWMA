import numpy as np
import argparse
import h5py
import time
import os
import pickle

import torch
import torch.nn.parallel
import torch.utils.data

from utils.model_supcon import PointNet_SupCon, PointNet_Classifier
from utils.dataset import TestDataset, ORGDataset
from utils.metrics_plots import classify_report, calculate_entire_data_average_metric
from utils.logger import create_logger


def load_test_data(args, logger, num_fold):
    """load train and validation data"""
    # load feature and label data
    val_dataset = ORGDataset(
        root=args.input_path,
        logger=logger,
        num_fold=num_fold,
        k=args.k_fold,
        split='val')

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.val_batch_size,
        shuffle=False, num_workers=int(args.num_workers))

    val_data_size = len(val_dataset)
    logger.info('The validation data size is:{}'.format(val_data_size))
    num_classes = len(val_dataset.label_names)
    logger.info('The number of classes is:{}'.format(num_classes))

    # load label names
    label_names = val_dataset.obtain_label_names()
    label_names_h5 = h5py.File(os.path.join(args.out_path, 'label_names.h5'), 'w')
    label_names_h5['y_names'] = label_names
    logger.info('The label names are: {}'.format(str(label_names)))

    return val_loader, label_names, num_classes


def contrastive_one_stage_eval_net(encoder_params, args, encoder_net, classifier_net, test_data_loader, label_names, script_name, logger, log_res_path, device):
    """Evaluate on one-stage model with contrastive learning"""
    logger.info('')
    logger.info('===================================')
    logger.info('')
    logger.info('{} Start multi-cluster prediction.'.format(script_name))

    output_prediction_report_path = os.path.join(log_res_path, 'entire_data_validation_results_best_{}.h5'.format(args.best_metric))
    # Load model
    start_time = time.time()
    with torch.no_grad():
        total_test_correct = 0
        test_labels_lst, test_predicted_lst= [], []
        critical_points_idx_matrix = None
        for j, data in (enumerate(test_data_loader, 0)):
            points, labels = data
            points = points.transpose(2, 1)
            labels = labels[:, 0]
            points, labels = points.to(device), labels.to(device)
            encoder_net, classifier_net = encoder_net.eval(), classifier_net.eval()
            # stage 2
            features, _, _, critical_points_idx = encoder_net.encoder(points)
            pred = classifier_net(features)
            _, pred_idx = torch.max(pred, dim=1)
            pred_idx = torch.where(pred_idx < 198, pred_idx, torch.tensor(198).to(device))
            correct = pred_idx.eq(labels.data).cpu().sum()
            # for calculating test accuracy
            total_test_correct += correct.item()
            # for calculating test weighted and macro metrics
            labels = labels.cpu().detach().numpy().tolist()
            test_labels_lst.extend(labels)
            pred_idx = pred_idx.cpu().detach().numpy().tolist()
            test_predicted_lst.extend(pred_idx)
            if args.analy_critical_points:
                if j == 0:
                    critical_points_idx_matrix = critical_points_idx.cpu().detach().numpy()
                else:
                    critical_points_idx_matrix = np.concatenate((critical_points_idx_matrix, critical_points_idx.cpu().detach().numpy()), axis=0)
    end_time = time.time()
    logger.info('The total time of prediction is:{} s'.format(round((end_time - start_time), 4)))
    logger.info('The test sample size is: {}'.format(len(test_predicted_lst)))
    label_names_str = [str(label_name) for label_name in label_names]
    classify_report(test_labels_lst, test_predicted_lst, label_names_str, logger, output_prediction_report_path, '{}_test'.format(args.best_metric))


def kfold_evaluate_one_stage_contrastive_model(encoder_params, args, device, script_name):
    log_res_path_base = os.path.join(args.out_path_base, 'NoStage1Model')
    fold_lst = encoder_params['fold_lst']
    for num_fold in fold_lst:
        num_fold = num_fold + 1
        log_res_path = os.path.join(log_res_path_base, str(num_fold))
        args.out_path = os.path.join(args.out_path_base, str(num_fold))
        # Record the training process and values
        try:
            os.makedirs(log_res_path)
        except OSError:
            pass
        # Record the training process and values
        logger = create_logger(log_res_path, 'evaluate_on_entire_data')
        logger.info('=' * 55)
        logger.info(args)
        logger.info('=' * 55)

        args.data_normalization = encoder_params['data_normalization']
        args.data_augmentation = encoder_params['data_augmentation']
        # load test data
        test_data_loader, label_names, num_class = load_test_data(args, logger, num_fold)

        # load model
        encoder = PointNet_SupCon(head=encoder_params['head_name'], feat_dim=encoder_params['encoder_feat_num'])
        classifer = PointNet_Classifier(num_classes=encoder_params['stage2_num_class']).to(device)

        # load weights
        encoder_weight_path_base = os.path.join(*args.out_path_base.split('/')[:-1])
        encoder_weight_path = os.path.join(encoder_weight_path_base, str(num_fold), 'epoch_{}_model.pth'.format(args.supcon_epoch))
        encoder.load_state_dict(torch.load(encoder_weight_path))
        classifier_weight_path = os.path.join(args.out_path, 'best_{}_model.pth'.format(args.best_metric))
        classifer.load_state_dict(torch.load(classifier_weight_path))

        # evaluation
        contrastive_one_stage_eval_net(encoder_params, args, encoder, classifer, test_data_loader, label_names, script_name, logger, log_res_path, device)

    # clean the logger
    logger.handlers.clear()

    # calculate the average performance
    calculate_entire_data_average_metric(log_res_path_base, len(fold_lst), args.best_metric, 'NoStage1Model')


