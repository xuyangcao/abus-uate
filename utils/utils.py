import os
import sys
import time
import torch
import shutil
import logging
import importlib
from medpy import metric
import numpy as np

def save_checkpoint(state, is_best, path, prefix, filename='checkpoint.pth.tar'):
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pth.tar')

def confusion(y_pred, y_true):
    '''
    get precision and recall
    '''
    y_pred = y_pred.float().view(-1) 
    y_true = y_true.float().view(-1)
    #print('y_pred.shape', y_pred.shape)
    #print('y_true.shape', y_true.shape)
    smooth = 1. 
    #y_pred_pos = np.clip(y_pred, 0, 1)
    y_pred_pos = y_pred
    y_pred_neg = 1 - y_pred_pos
    #y_pos = np.clip(y_true, 0, 1)
    y_pos = y_true
    y_neg = 1 - y_true

    tp = torch.dot(y_pos, y_pred_pos)
    fp = torch.dot(y_neg, y_pred_pos)
    fn = torch.dot(y_pos, y_pred_neg)

    prec = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)

    return prec, recall

def gaussian_noise(x, batchsize, input_shape=(3, 224, 224), std=0.03):
    noise = torch.zeros(x.shape)
    noise.data.normal_(0, std)
    return x + noise

def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


def get_metrics(pred, gt, voxelspacing=(0.5, 0.5, 0.5)):
    r""" 
    Get statistic metrics of segmentation

    These metrics include: Dice, Jaccard, Hausdorff Distance, 95% Hausdorff Distance, 
    and Average surface distance(ASD) metric.

    If the prediction result is 0s, we set hd, hd95, asd 10.0 to avoid errors.

    Parameters:
    -----------
    pred: 3D numpy ndarray
        binary prediction results 

    gt: 3D numpy ndarray
        binary ground truth

    voxelspacing: tuple of 3 floats. default: (0.5, 0.5, 0.5)
        voxel space of 3D image

    Returns:
    --------
    metrics: dict of 5 metrics 
        dict{dsc, jc, hd, hd95, asd}
    """

    dsc = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    vs = np.sum(gt) * voxelspacing[0] * voxelspacing[1] * voxelspacing[2]
    precision = metric.binary.precision(pred, gt)
    recall = metric.binary.recall(pred, gt)

    if np.sum(pred) == 0:
        print('=> prediction is 0s! ')
        hd = 32.
        hd95 = 32.
        asd = 10.
    else:
        hd = metric.binary.hd(pred, gt, voxelspacing=voxelspacing)
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=voxelspacing)
        asd = metric.binary.asd(pred, gt, voxelspacing=voxelspacing)

    metrics = {'dsc': dsc, 'jc': jc, 'hd': hd, 'hd95': hd95, 'asd': asd, 
                'precision':precision, 'recall':recall, 'vs': vs} 
    return metrics 

def get_dice(pred, gt):
    dice = metric.binary.dc(pred, gt)

    return dice


def load_config(file_name):
    r"""
    load configuration file as a python module

    Args:
        file_name: configuration file name

    Returns:
        a loaded network module            
    """

    dir_name = os.path.dirname(file_name)
    base_name = os.path.basename(file_name)
    module_name, _ = os.path.splitext(base_name)
    #print('dir_name: ', dir_name)
    #print('base_name: ', base_name)
    #print('module_name: ', module_name)

    os.sys.path.append(dir_name)
    config = importlib.import_module(module_name)
    if module_name in sys.modules:
        importlib.reload(config)
    del os.sys.path[-1]

    return config.cfg


if __name__ == "__main__":
    cfg = load_config('../config/config.py')
    print(cfg)
