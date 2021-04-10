import torch
import glob
import os
import sys

from torchvision import transforms
from torchvision.transforms import functional as F
#import cv2
from PIL import Image
# import pandas as pdSegmentationMapOnImage
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
#from utils import get_label_info, one_hot_it
import random
import skimage.io as io
# from utils.config import DefaultConfig

def augmentation():
    # augment images with spatial transformation: Flip, Affine, Rotation, etc...
    # see https://github.com/aleju/imgaug for more details
    pass

def augmentation_pixel():
    # augment images with pixel intensity transformation: GaussianBlur, Multiply, etc...
    pass

class OCT(torch.utils.data.Dataset):
    Unlabelled=[0,0,0]
    Aorta = [255,255,255]
    COLOR_DICT = np.array([Unlabelled,Aorta])
    def __init__(self, dataset_path,scale=(256,448), mode='train'):
        super().__init__()
        self.mode = mode
        self.img_path=dataset_path+'/'+mode+'/img'
        self.mask_path=dataset_path+'/'+mode+'/mask'
        self.image_lists,self.label_lists=self.read_list(self.img_path)
        self.flip =iaa.SomeOf((1,4),[
             iaa.Fliplr(0.5),
             iaa.Flipud(0.1),
             iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
             iaa.Affine(rotate=(-5, 5),
                        scale={"x": (0.9, 1.1), "y": (0.8, 1.2)}),
             iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(3, 5)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 5)), # blur image using local medians with kernel sizes between 2 and 7
                ]),
             iaa.ContrastNormalization((0.5, 1.5))], random_order=True)
        # resize
        self.resize_label = transforms.Resize(scale, Image.NEAREST)
        self.resize_img = transforms.Resize(scale, Image.BILINEAR)
        # normalization
        self.to_tensor = transforms.ToTensor()


    def __getitem__(self, index):
        # load image and crop
        img = Image.open(self.image_lists[index])
        img = self.resize_img(img)
        img = np.array(img)
        length=len(self.image_lists)
        labels = self.label_lists[index]
        #load label
        if self.mode !='test':
            label_ori = Image.open(self.label_lists[index]).convert("RGB")
            label_ori = self.resize_label(label_ori)
            label_ori = np.array(label_ori)

            label=np.ones(shape=(label_ori.shape[0],label_ori.shape[1]),dtype=np.uint8)

            #convert RGB  to one hot
            
            for i in range(len(self.COLOR_DICT)):
                equality = np.equal(label_ori, self.COLOR_DICT[i])
                class_map = np.all(equality, axis=-1)
                label[class_map]=i

            # augment image and label
            if self.mode == 'train' or self.mode == 'train_val' :
                seq_det = self.flip.to_deterministic()#固定变换
                segmap = ia.SegmentationMapsOnImage(label, shape=label.shape, nb_classes=4)
                img = seq_det.augment_image(img)
                label = seq_det.augment_segmentation_maps([segmap])[0].get_arr().astype(np.uint8)

            label_img=torch.from_numpy(label.copy()).float()
            if self.mode == 'val':
                 img_num=len(os.listdir(os.path.dirname(labels)))
                 labels=label_img,img_num
            else:
                 labels=label_img
            #labels = label_img

        imgs=img.transpose(2,0,1)/255.0
        img = torch.from_numpy(imgs.copy()).float()#self.to_tensor(img.copy()).float()
        return img, labels

    def __len__(self):
        return len(self.image_lists)
    def read_list(self,image_path):
        fold = os.listdir(image_path)
        os.listdir()
        img_list=[]
        if self.mode=='train':
            fold_r=fold
            # fold_r.remove('f'+str(k_fold_test))# remove testdata
            for item in fold_r:
                img_list.append(os.path.join(image_path,item))
            label_list = [x.replace('img','mask') for x in img_list]
            label_list=[x[:-4]+'_mask.png' for x in label_list]

        elif self.mode=='val':
            fold_r = fold
            for item in fold_r:
                img_list.append(os.path.join(image_path, item))
            label_list = [x.replace('img', 'mask') for x in img_list]
            label_list = [x[:-4] + '_mask.png' for x in label_list]


        elif self.mode=='test':
            fold_r = fold
            # fold_r.remove('f'+str(k_fold_test))# remove testdata
            for item in fold_r:
                img_list.append(os.path.join(image_path, item))
            label_list = [x.replace('img', 'mask') for x in img_list]
            label_list = [x[:-4] + '_mask.png' for x in label_list]
                
        return img_list,label_list







if __name__ == '__main__':
   data = OCT(r'/root/qiu/dataset/data_med4', (512, 512),mode='train')
   print(data.__getitem__(0))


   

