# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 10:59:53 2018

"""
import numpy as np
import os 
from PIL import Image
size = (256, 192)
dice  =0
def evaluate(path_predict):
    path_true=os.path.join(r'E:\dataset\skin1\skin\TestDataset/test\masks')
    TP=FPN=0
    Jaccard=[]
    ds = []
    acc_arr = []
    sensitivitys = []
    recall_arr=[]
    precisions_arr=[]
    for roots, dirs, files in os.walk(path_predict):
        if files:
            for file in files:
                pre_file_path=os.path.join(roots, file)

                true_file_path=os.path.join(path_true, file)
                img_pre = np.array(Image.open(pre_file_path).convert("L"))
                img_pre = img_pre / 255
                img_true = Image.open(true_file_path).convert("L")
                img_true = np.array(img_true)
                img_true = img_true / 255
                img_true = (img_true>0.5)
                img_pre = (img_pre>0.5)
                img_true = np.array(img_true).astype(dtype=np.int)
                img_pre = np.array(img_pre).astype(dtype=np.int)

                inter = np.sum(img_pre*img_true)
                unite = np.sum(img_pre)+np.sum(img_true)
                TP = TP+inter
                FPN = FPN + unite
                ds.append(2*inter/unite)
                single_I=inter
                single_U=unite-single_I
                precisions_arr.append(inter/(img_pre.sum()+0.000001))
                recall_arr.append(inter/(img_true.sum()+0.000001))
                Jaccard.append(single_I/single_U)
                sensitivity = inter/(np.sum(img_true)+0.000001)
                sensitivitys.append(sensitivity)
                acc = (img_true == img_pre).sum()/(size[0]*size[1])
                acc_arr.append(acc)
    print("recall", sum(recall_arr)/len(recall_arr))
    print("precision", sum(precisions_arr)/len(recall_arr))
    print("mean_dice", sum(ds)/len(ds))
    print("acc=", sum(acc_arr)/len(acc_arr))
    print("sensitivity=", sum(sensitivitys)/len(sensitivitys))
    dice = 2*TP/FPN
    print('TP:',TP)
    print('FPN:',FPN)
    print("DICE",dice)
    print('glob_Jaccard',TP/(FPN-TP))
    print('single_Jaccard',sum(Jaccard)/len(Jaccard))
    return dice


if __name__ == '__main__':
    path = r'E:\dataset\skin1\skin\TestDataset\test\output'
    evaluate(path)