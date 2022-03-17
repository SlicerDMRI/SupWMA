import numpy as np
import h5py
import os
import sys
import copy
import torch
import matplotlib.ticker as mtick
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_recall_fscore_support
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN


sys.path.append('..')
from utils.logger import create_logger
from utils.funcs import round_decimal_percentage, round_decimal


def calculate_prec_recall_f1(labels_lst, predicted_lst):
    # Beta: The strength of recall versus precision in the F-score. beta == 1.0 means recall and precision are equally important, that is F1-score
    mac_precision, mac_recall, mac_f1, _ = precision_recall_fscore_support(y_true=labels_lst, y_pred=predicted_lst, beta=1.0, average='macro')
    return mac_precision, mac_recall, mac_f1


def classify_report(labels_lst, predicted_lst, label_names, logger, out_path, metric_name):
    """Generate classification performance report"""
    cls_report = classification_report(y_true=labels_lst, y_pred=predicted_lst, digits=5, target_names=label_names)
    logger.info('=' * 55)
    logger.info('Best {} classification report:\n{}'.format(metric_name, cls_report))
    logger.info('=' * 55)
    logger.info('\n')

    if 'test' in metric_name:
        test_res = h5py.File(out_path, "w")
        test_res['val_predictions'] = predicted_lst
        test_res['val_labels'] = labels_lst
        test_res['label_names'] = label_names
        test_res['classification_report'] = cls_report
    else:
        val_res = h5py.File(os.path.join(out_path, 'entire_data_validation_results_best_{}.h5'.format(metric_name)), "w")
        val_res['val_predictions'] = predicted_lst
        val_res['val_labels'] = labels_lst
        val_res['label_names'] = label_names
        val_res['classification_report'] = cls_report


def per_class_metric(labels_lst, predicted_lst, label_names, val_data_size, logger, out_path, metric_name):
    """Analysis for each class metric and its metric"""
    cls_report_dict = classification_report(y_true=labels_lst, y_pred=predicted_lst, digits=5,
                                            target_names=label_names, output_dict=True)
    ratio_lst = []
    precision_lst = []
    recall_lst = []
    f1_lst = []
    for key in cls_report_dict.keys():
        if 'b' in key:
            ratio = cls_report_dict[key]['support'] / float(val_data_size)
            cls_report_dict[key]['ratio'] = ratio
            ratio_lst.append(ratio)
            precision_lst.append(cls_report_dict[key]['precision'])
            recall_lst.append(cls_report_dict[key]['recall'])
            f1_lst.append(cls_report_dict[key]['f1-score'])
    np.save(os.path.join(out_path, 'cls_report_dict_best_{}.npy'.format(metric_name)), cls_report_dict)

    # comparison figures
    ratio_lst = [ratio * 100 for ratio in ratio_lst]
    fig = plt.figure(figsize=(30, 10))
    # precision vs ratio
    ax1 = plt.subplot(1, 3, 1)
    # recall vs ratio
    plt.scatter(ratio_lst[:-1], precision_lst[:-1], color='r')
    plt.xlabel('The amount ratio of streamlines (%)', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('(No "other fiber") The relationship between precision and the amount of streamlines', fontsize=12)
    fmt = '%.2f%%'  # ticks format, round to two decimal places
    xticks = mtick.FormatStrFormatter(fmt)
    ax1.xaxis.set_major_formatter(xticks)
    plt.grid()

    ax2 = plt.subplot(1, 3, 2)
    plt.scatter(ratio_lst[:-1], recall_lst[:-1], color='b')
    plt.xlabel('The amount ratio of streamlines (%)', fontsize=12)
    plt.ylabel('Recall', fontsize=12)
    plt.title('(No "other fiber") The relationship between recall and the amount of streamlines', fontsize=12)
    fmt = '%.2f%%'  # ticks format, round to two decimal places
    xticks = mtick.FormatStrFormatter(fmt)
    ax2.xaxis.set_major_formatter(xticks)
    plt.grid()
    # f1-score vs ratio
    ax3 = plt.subplot(1, 3, 3)
    plt.scatter(ratio_lst[:-1], f1_lst[:-1], color='g')
    plt.xlabel('The amount ratio of streamlines (%)', fontsize=12)
    plt.ylabel('f1-score', fontsize=12)
    plt.title('(No "other fiber") The relationship between f1-score and the amount of streamlines', fontsize=12)
    fmt = '%.2f%%'  # ticks format, round to two decimal places
    xticks = mtick.FormatStrFormatter(fmt)
    ax3.xaxis.set_major_formatter(xticks)
    plt.grid()
    plt.savefig(os.path.join(out_path, 'metric_quantity_comparison_best_{}.png'.format(metric_name)))


def process_curves(epoch, train_loss_lst, val_loss_lst, train_acc_lst, val_acc_lst,
                    train_precision_lst, val_precision_lst, train_recall_lst, val_recall_lst,
                    train_f1_lst, val_f1_lst, best_acc, best_acc_epoch, best_f1_mac, best_f1_epoch, out_path):
    """Generate training curves"""
    epoch_lst = range(1, epoch + 1)
    fig, axes = plt.subplots(2, 3, figsize=(30, 20))
    # loss
    axes[0, 0].plot(epoch_lst, train_loss_lst, '-', color='r', label='train loss')
    axes[0, 0].plot(epoch_lst, val_loss_lst, '-', color='b', label='val loss')
    axes[0, 0].set_title('Loss Curve', fontsize=15)
    axes[0, 0].set_xlabel('epochs', fontsize=12)
    axes[0, 0].set_ylabel('loss', fontsize=12)
    axes[0, 0].grid()
    axes[0, 0].legend()
    axes[0, 0].set_xticks(np.arange(1, len(epoch_lst) + 1, 5.0))

    # accuracy
    axes[0, 1].plot(epoch_lst, train_acc_lst, '-', color='r', label='train accuracy')
    axes[0, 1].plot(epoch_lst, val_acc_lst, '-', color='b', label='val accuracy')
    axes[0, 1].set_title('Accuracy Curve', fontsize=15)
    axes[0, 1].set_xlabel('epochs', fontsize=12)
    axes[0, 1].set_ylabel('accuracy', fontsize=12)
    axes[0, 1].grid()
    axes[0, 1].legend()
    axes[0, 1].set_xticks(np.arange(1, len(epoch_lst) + 1, 5.0))

    # macro precision
    axes[0, 2].plot(epoch_lst, train_precision_lst, '-', color='r', label='train precision')
    axes[0, 2].plot(epoch_lst, val_precision_lst, '-', color='b', label='val precision')
    axes[0, 2].set_title('Precision (marco) Curve', fontsize=15)
    axes[0, 2].set_xlabel('epochs', fontsize=12)
    axes[0, 2].set_ylabel('precision', fontsize=12)
    axes[0, 2].grid()
    axes[0, 2].legend()
    axes[0, 2].set_xticks(np.arange(1, len(epoch_lst) + 1, 5.0))

    # macro recall
    axes[1, 0].plot(epoch_lst, train_recall_lst, '-', color='r', label='train recall')
    axes[1, 0].plot(epoch_lst, val_recall_lst, '-', color='b', label='val recall')
    axes[1, 0].set_title('Recall (marco) Curve', fontsize=15)
    axes[1, 0].set_xlabel('epochs', fontsize=12)
    axes[1, 0].set_ylabel('recall', fontsize=12)
    axes[1, 0].grid()
    axes[1, 0].legend()
    axes[1, 0].set_xticks(np.arange(1, len(epoch_lst) + 1, 5.0))

    # macro f1
    axes[1, 1].plot(epoch_lst, train_f1_lst, '-', color='r', label='train f1')
    axes[1, 1].plot(epoch_lst, val_f1_lst, '-', color='b', label='val f1')
    axes[1, 1].set_title('F1-score (marco) Curve', fontsize=15)
    axes[1, 1].set_xlabel('epochs', fontsize=12)
    axes[1, 1].set_ylabel('f1-score', fontsize=12)
    axes[1, 1].grid()
    axes[1, 1].legend()
    axes[1, 1].set_xticks(np.arange(1, len(epoch_lst) + 1, 5.0))

    # accuracy,  macro precision, macro recall, macro f1
    axes[1, 2].plot(epoch_lst, val_acc_lst, '-', color='r', label='accuracy')
    axes[1, 2].scatter(best_acc_epoch, best_acc, c='r', marker='8', label='best accuracy')
    axes[1, 2].plot(epoch_lst, val_precision_lst, '-', color='g', label='precision (macro)')
    axes[1, 2].plot(epoch_lst, val_recall_lst, '-', color='b', label='recall (macro)')
    axes[1, 2].plot(epoch_lst, val_f1_lst, '-', color='y', label='f1 (macro)')
    axes[1, 2].scatter(best_f1_epoch, best_f1_mac, c='y', marker='P', label='best f1 (macro)')
    axes[1, 2].set_title('Metric Comparison Curve', fontsize=15)
    axes[1, 2].set_xlabel('epochs', fontsize=12)
    axes[1, 2].set_ylabel('metrics', fontsize=12)
    axes[1, 2].grid()
    axes[1, 2].legend()
    axes[1, 2].set_xticks(np.arange(1, len(epoch_lst) + 1, 5.0))
    plt.savefig(os.path.join(out_path, 'train_validation_process_analysis.png'))


def best_swap(metric, epoch, net, labels_lst, predicted_lst):
    best_metric = metric
    best_epoch = epoch
    best_wts = copy.deepcopy(net.state_dict())
    best_labels_lst = labels_lst
    best_pred_lst = predicted_lst
    return best_metric, best_epoch, best_wts, best_labels_lst, best_pred_lst


def save_best_weights(net, best_wts, out_path, metric_name, epoch, metric_value, logger):
    net.load_state_dict(best_wts)
    torch.save(net.state_dict(), '{}/best_{}_model.pth'.format(out_path, metric_name))
    logger.info('The model with best {} is saved: epoch {}, {} {}'.format(metric_name, epoch, metric_name, metric_value))


def _metric_across_clusters( cls_report_dict, label_names):
    """Calculate mean and standard deviation for clusters of each fold"""
    precision_across_clusters = np.zeros(len(label_names))
    recall_across_clusters = np.zeros(len(label_names))
    f1_across_clusters = np.zeros(len(label_names))
    for idx_cluster, label in enumerate(label_names):
        # If we wants to report percentage of variance, must multiply 100 here!
        precision_across_clusters[idx_cluster] = cls_report_dict[label]['precision'] * 100
        recall_across_clusters[idx_cluster] = cls_report_dict[label]['recall'] * 100
        f1_across_clusters[idx_cluster] = cls_report_dict[label]['f1-score'] * 100
    accuracy = cls_report_dict['accuracy'] * 100
    macro_precision = np.mean(precision_across_clusters)
    std_precision = np.std(precision_across_clusters)
    macro_recall = np.mean(recall_across_clusters)
    std_recall = np.std(recall_across_clusters)
    macro_f1 = np.mean(f1_across_clusters)
    std_f1 = np.std(f1_across_clusters)

    return [accuracy, macro_precision, std_precision, macro_recall, std_recall, macro_f1, std_f1]


def _mean_std_across_folds(accuracy_array, precision_array, recall_array, f1_array,
                           h5_base_path, num_average_files, metric_name, logger):
    """Calculate mean and standard deviation for folds"""
    avg_acc = round_decimal(np.mean(accuracy_array), decimal=3)
    avg_precision = round_decimal(np.mean(precision_array[:, 0]), decimal=3)
    avg_recall = round_decimal(np.mean(recall_array[:, 0]), decimal=3)
    avg_f1 = round_decimal(np.mean(f1_array[:, 0]), decimal=3)

    std_precision = round_decimal(np.mean(precision_array[:, 1]), decimal=3)
    std_recall = round_decimal(np.mean(recall_array[:, 1]), decimal=3)
    std_f1 = round_decimal(np.mean(f1_array[:, 1]), decimal=3)

    logger.info('The number of experiment implementations is {}'.format(num_average_files))
    logger.info('Use the weight with best {} for each fold'.format(metric_name))
    logger.info('='*55)

    logger.info('The average accuracy for {} is {} %\n'.format(h5_base_path, avg_acc))
    logger.info('The average macro precision for {} is {} % and standard deviation is {} %\n'.format(h5_base_path, avg_precision, std_precision))
    logger.info('The average macro recall for {} is {} % and standard deviation is {} %\n'.format(h5_base_path, avg_recall, std_recall))
    logger.info('The average macro f1 for {} is {} % and standard deviation is {} %\n'.format(h5_base_path, avg_f1, std_f1))
    logger.info('='*55)


def calculate_average_metric(h5_base_path, num_average_files, metric_name, redistribute_class):
    accuracy_array = np.zeros(num_average_files)
    precision_array = np.zeros((num_average_files, 2))
    recall_array = np.zeros((num_average_files, 2))
    f1_array = np.zeros((num_average_files, 2))
    logger = create_logger(h5_base_path, 'MeanStd_Results')
    # logger.info('Not calculating stage 1 or stage 2')
    if redistribute_class:
        logger.info('Redistribute class')
    for i in range(num_average_files):
        if redistribute_class:
            h5_path = os.path.join(h5_base_path, str(i + 1), 'validation_results_best_{}_redistribute.h5'.format(metric_name))
        else:
            h5_path = os.path.join(h5_base_path, str(i+1), 'entire_data_validation_results_best_{}.h5'.format(metric_name))
        results = h5py.File(h5_path, 'r')
        labels_lst = results['val_labels']
        predicted_lst = results['val_predictions']
        label_names = results['label_names']
        cls_report_dict = classification_report(y_true=labels_lst, y_pred=predicted_lst, digits=7,
                                                target_names=label_names, output_dict=True)

        # calculate mean and standard deviation across clusters for each fold
        # metric_results_lst: [accuracy, macro_precision, std_precision, macro_recall, std_recall, macro_f1, std_f1]
        results_per_fold = _metric_across_clusters(cls_report_dict, label_names)
        accuracy_array[i] = results_per_fold[0]
        precision_array[i, :] = [results_per_fold[1], results_per_fold[2]]
        recall_array[i, :] = [results_per_fold[3], results_per_fold[4]]
        f1_array[i, :] = [results_per_fold[5], results_per_fold[6]]

    logger.info('The number of classes is {}'.format(len(label_names)))
    _mean_std_across_folds(accuracy_array, precision_array, recall_array, f1_array,
                           h5_base_path, num_average_files, metric_name, logger)


def gen_199_classify_report(labels_lst, predicted_lst, label_names, logger, out_path, metric_name):
    """Redistribute classes to 198 gt swm and 1 other"""
    label_names = label_names[:198]
    label_names.append('all_others')
    labels_lst = np.asarray(labels_lst)
    predicted_lst = np.asarray(predicted_lst)
    labels_lst_mask = np.where(labels_lst > 197)
    labels_lst[labels_lst_mask] = 198
    predicted_lst_mask = np.where(predicted_lst > 197)
    predicted_lst[predicted_lst_mask] = 198
    cls_report = classification_report(y_true=labels_lst, y_pred=predicted_lst, digits=5, target_names=label_names)
    if logger is not None:
        logger.info('=' * 55)
        logger.info('Best {} redistribute classification report:\n{}'.format(metric_name, cls_report))
        logger.info('=' * 55)
        logger.info('\n')

    val_res = h5py.File(os.path.join(out_path, 'validation_results_best_{}_redistribute.h5'.format(metric_name)), "w")
    val_res['val_predictions'] = predicted_lst
    val_res['val_labels'] = labels_lst
    val_res['label_names'] = label_names
    val_res['classification_report'] = cls_report


