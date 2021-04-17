#coding=gbk
"""
Created on Wed Sep 19 10:59:53 2018

"""
import numpy as np
import os 
from PIL import Image
path_true=r'/root/qiu/dataset/data_med4/test/mask'
path_predict=r'/root/qiu/dataset/data_med4/test/predict'
#path_true=r'E:\dataset\data_med4\test\mask'
#path_predict=r'E:\dataset\data_med4\test\predict'
TP=FPN=0
Jaccard=[]
from torchvision import  transforms
scale=(256,448)
resize_label = transforms.Resize(scale, Image.NEAREST)
dices=[]


def eval_multi_seg(predict, target, num_classes):
    # pred_seg=torch.argmax(torch.exp(predict),dim=1).int()
    pred_seg = predict # n h w
    label_seg = target  # n h w
    assert (pred_seg.shape == label_seg.shape)
    acc =0
#    acc = (pred_seg == label_seg).sum() / (
 #               pred_seg.shape[0] * pred_seg.shape[1] * pred_seg.shape[2])  # acc 就是所有相同的像素值占总像素的大小

    # Dice = []
    # Precsion = []
    # Jaccard = []
    # Sensitivity=[]
    # Specificity=[]

    # n = pred_seg.shape[0]
    Dice = []
    True_label = []
    for classes in range(1, num_classes):  # 循环遍历说有的类型  没有遍历0 的原因是 0 是背景 说以就不便利
        overlap = ((pred_seg == classes) * (label_seg == classes)).sum()
        union = (pred_seg == classes).sum() + (label_seg == classes).sum()
        Dice.append((2 * overlap + 0.1) / (union + 0.1))
        True_label.append((label_seg == classes).sum())

    return Dice, True_label, acc
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

         #   TP = TP+np.sum(img_pre*img_true)
          #  FPN = FPN +np.sum(img_pre)+np.sum(img_true)
            #overlap=((img_pre==1)*(img_true==1)).sum()
           # union=(img_pre==1).sum()+(img_true==1).sum()
            overlap = np.sum(img_pre*img_true)
            union =  np.sum(img_pre)+np.sum(img_true)
            dices.append((2*overlap+0.01)/(union+0.01))
            #Di,True_label,acc = eval_multi_seg(img_pre,img_true,2)
            #dices.append(Di[0])
            single_I=np.sum(img_pre*img_true)
            single_U=np.sum(img_pre)+np.sum(img_true)-single_I
            Jaccard.append(single_I/single_U)



#dice = 2*TP/FPN
dice  = sum(dices)/len(dices)
print('TP:',TP)
print('FPN:',FPN)           
print("DICE",dice)
#print('glob_Jaccard',TP/(FPN-TP))
print('single_Jaccard',sum(Jaccard)/len(Jaccard))
            
