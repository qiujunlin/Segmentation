# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 10:59:53 2018

"""

import numpy as np
import os 
from PIL import Image
path_true=r'/root/qiu/dataset/data_med4/test/mask'
path_predict=r'/root/qiu/dataset/data_med4/test/predict'
TP=FPN=0
Jaccard=[]
from torchvision import  transforms
scale=(256,448)
resize_label = transforms.Resize(scale, Image.NEAREST)
for roots,dirs,files in os.walk(path_predict):
    if files:
#        dice=[]
#        num=0
        for file in files:
#            num=num+1
            pre_file_path=os.path.join(roots,file)
            true_file_path=os.path.join(path_true,file[:-11]+'mask.png')
            img_pre = Image.open(pre_file_path).convert("L")
            img_pre = np.array(img_pre)
            img_pre[img_pre==255]=1
            img_true = Image.open(true_file_path).convert("L")
            img_true =  resize_label(img_true)
            img_true = np.array(img_true)
            img_true[img_true==255]=1
#            print(img_pre.shape)
#            print(img_true.shape)
#            TP = TP+np.sum(np.array(img_pre,dtype=np.int32)&np.array(img_true,dtype=np.int32))
#            FPN = FPN +np.sum(np.array(img_pre,dtype=np.int32)|np.array(img_true,dtype=np.int32))
            TP = TP+np.sum(img_pre*img_true)
            FPN = FPN +np.sum(img_pre)+np.sum(img_true)
            single_I=np.sum(img_pre*img_true)
            single_U=np.sum(img_pre)+np.sum(img_true)-single_I
            Jaccard.append(single_I/single_U)


dice = 2*TP/FPN
print('TP:',TP)
print('FPN:',FPN)           
print("DICE",dice)
print('glob_Jaccard',TP/(FPN-TP))
print('single_Jaccard',sum(Jaccard)/len(Jaccard))
            
