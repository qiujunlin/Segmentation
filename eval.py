import os
import argparse
import tqdm
import yaml
import sys

import numpy as np

from PIL import Image
from tabulate import tabulate
from easydict import EasyDict as ed
from torch.utils.data import DataLoader
import  torch
from config.config import DefaultConfig
from dataset.Dataset import TestDataset

from utils.metrics import evaluate
from utils.metrics import Metrics


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/PraNet_Res2Net.yaml')
    return parser.parse_args()


def eval():


    args = DefaultConfig()
    print('#' * 20, 'Start Evaluation', '#' * 20)
    for dataset in tqdm.tqdm(args.testdataset, total=len(args.testdataset), position=0,
                             bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}'):
        pred_path = 'E:/dataset/data/TestDataset/{}/output/'.format(dataset)
        gt_path = 'E:\dataset/data/TestDataset/{}/masks/'.format(dataset)
        preds = os.listdir(pred_path)
        gts = os.listdir(gt_path)
        total_batch =  len(preds)
        # metrics_logger initialization
        metrics = Metrics(['recall', 'specificity', 'precision', 'F1', 'F2',
                           'ACC_overall', 'IoU_poly', 'IoU_bg', 'IoU_mean', 'Dice'])

        for i, sample in tqdm.tqdm(enumerate(zip(preds, gts)), desc=dataset + ' - Evaluation', total=len(preds),
                                   position=1, leave=False, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}'):
            pred, gt = sample
            assert os.path.splitext(pred)[0] == os.path.splitext(gt)[0]

            pred_mask = np.array(Image.open(os.path.join(pred_path, pred)))
            gt_mask = np.array(Image.open(os.path.join(gt_path, gt)))
            if len(pred_mask.shape) != 2:
                pred_mask = pred_mask[:, :, 0]
            if len(gt_mask.shape) != 2:
                gt_mask = gt_mask[:, :, 0]

            assert pred_mask.shape == gt_mask.shape
            gt_mask = gt_mask.astype(np.float64) / 255
            pred_mask = pred_mask.astype(np.float64) / 255

            gt_mask = torch.from_numpy(gt_mask)
            pred_mask =  torch.from_numpy(pred_mask)
            _recall, _specificity, _precision, _F1, _F2, \
            _ACC_overall, _IoU_poly, _IoU_bg, _IoU_mean, _Dice = evaluate(pred_mask, gt_mask, 0.5)

            metrics.update(recall=_recall, specificity=_specificity, precision=_precision,
                           F1=_F1, F2=_F2, ACC_overall=_ACC_overall, IoU_poly=_IoU_poly,
                           IoU_bg=_IoU_bg, IoU_mean=_IoU_mean, Dice=_Dice
                           )
        metrics_result = metrics.mean(total_batch)
        print("Test Result:")
        print('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, '
              'ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f, Dice:%.4f'
              % (metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'],
                 metrics_result['F1'], metrics_result['F2'], metrics_result['ACC_overall'],
                 metrics_result['IoU_poly'], metrics_result['IoU_bg'], metrics_result['IoU_mean'],
                 metrics_result['Dice']))



if __name__ == "__main__":

    eval()

