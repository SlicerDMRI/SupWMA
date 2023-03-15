import numpy as np
import h5py
import time
import os
import pickle

import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data

from utils.logger import create_logger
from utils.model import PointNetCls
from utils.model_supcon import PointNet_SupCon, PointNet_Classifier
from utils.dataset import ORGDataset
from utils.metrics_plots import classify_report, calculate_entire_data_average_metric
from utils.funcs import makepath


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
        shuffle=True, num_workers=int(args.num_workers))

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


def contrastive_two_stage_eval_net(stage1_params, encoder_params, args, stage1_net, stage2_encoder_net, stage2_classifer_net,
                                   test_data_loader, label_names, script_name, logger, log_res_path, device):
    """perform predition of two-stage model with contrastive loss"""
    logger.info('')
    logger.info('===================================')
    logger.info('')
    logger.info('{} Start multi-cluster prediction.'.format(script_name))

    output_prediction_report_path = os.path.join(log_res_path, 'entire_data_validation_results_best_{}.h5'.format(args.best_metric))
    # Load model
    start_time = time.time()
    with torch.no_grad():
        total_test_correct = 0
        test_labels_lst, test_predicted_lst, test_swm_labels_lst = [], [], []
        encoder_swm_features_array = None
        tot_swm_points = None
        for j, data in (enumerate(test_data_loader, 0)):
            points, labels = data
            points = points.transpose(2, 1)
            labels = labels[:, 0]
            points, labels = points.to(device), labels.to(device)
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
                if args.analy_fiber_representation:
                    stage2_encoder_feat = features.cpu().detach()
                    test_swm_labels_lst.extend(labels[stage1_swm_mask].cpu().detach())
                    if encoder_swm_features_array is None:
                        encoder_swm_features_array = stage2_encoder_feat
                    else:
                        encoder_swm_features_array = np.concatenate((encoder_swm_features_array, stage2_encoder_feat),
                                                                    axis=0)

                stage2_pred = stage2_classifer_net(features)
                _, stage2_pred_idx = torch.max(stage2_pred, dim=1)
                pred_idx[stage1_swm_mask] = torch.where(stage2_pred_idx < 198, stage2_pred_idx,
                                                        torch.tensor(198).to(device))

            # entire data
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
    pred_time = end_time - start_time
    logger.info('The total time of prediction is:{} s'.format(round((pred_time), 4)))
    logger.info('The test sample size is: {}'.format(len(test_predicted_lst)))
    label_names_str = [str(label_name) for label_name in label_names]
    classify_report(test_labels_lst, test_predicted_lst, label_names_str, logger, output_prediction_report_path, '{}_test'.format(args.best_metric))

    return pred_time




def kfold_evaluate_two_stage_contrastive_model(stage1_params, encoder_params, args, device, script_name):
    log_res_path_base = os.path.join(args.out_path_base, args.stage1_weight_path_base.split('/')[-1])
    fold_lst = encoder_params['fold_lst']
    total_prediction_time = 0
    for num_fold in fold_lst:
        num_fold = num_fold + 1
        log_res_path = os.path.join(log_res_path_base, str(num_fold))
        args.out_path = os.path.join(args.out_path_base, str(num_fold))
        # Record the training process and values
        makepath(log_res_path)
        # Record the training process and values
        logger = create_logger(log_res_path, 'evaluate_on_entire_data')
        logger.info('=' * 55)
        logger.info(args)
        logger.info('=' * 55)
        
        # load test data
        test_data_loader, label_names, num_class = load_test_data(args, logger, num_fold)

        # load model
        # model setting
        stage1_model = PointNetCls(k=stage1_params['stage1_num_class']).to(device)
        stage2_encoder = PointNet_SupCon(head=encoder_params['head_name'], feat_dim=encoder_params['encoder_feat_num']).to(device)
        stage2_classifer = PointNet_Classifier(num_classes=encoder_params['stage2_num_class']).to(device)

        # load weights
        if args.stage1_weight_path_base == '':
            raise NotImplementedError('The stage1 weight is required')
        else:
            stage1_weight_path = os.path.join(args.stage1_weight_path_base, str(num_fold), 'best_{}_model.pth'.format(args.best_metric))
            stage1_model.load_state_dict(torch.load(stage1_weight_path))
            encoder_weight_path_base = os.path.join(*args.out_path_base.split('/')[:-1])
            encoder_weight_path = os.path.join(encoder_weight_path_base, str(num_fold), 'epoch_{}_model.pth'.format(args.supcon_epoch))
            stage2_encoder.load_state_dict(torch.load(encoder_weight_path))
            classifier_weight_path = os.path.join(args.out_path, 'best_{}_model.pth'.format(args.best_metric))
            stage2_classifer.load_state_dict(torch.load(classifier_weight_path))

        # evaluation        
        prediction_time = contrastive_two_stage_eval_net(stage1_params, encoder_params, args, stage1_model, stage2_encoder, stage2_classifer,
                                        test_data_loader, label_names, script_name, logger, log_res_path, device)
        total_prediction_time += prediction_time
    
    # clean the logger
    logger.handlers.clear()

    if len(fold_lst) == 5:
        # calculate the average performance
        stage1_path = args.stage1_weight_path_base.split('/')[-1]
        calculate_entire_data_average_metric(log_res_path_base, len(fold_lst), args.best_metric, stage1_path)
        logger.info("The total prediction time for {} fold(s) is {} s".format(len(fold_lst), round(total_prediction_time,4)))



