import torch
import glob
import os
import sys
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as F
#import cv2
from PIL import Image
import random

class Dataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path,scale=(352,352),augmentations = True,hasEdg =False):
        super().__init__()
        self.augmentations = augmentations
        self.img_path=dataset_path+'/images/'
        self.mask_path=dataset_path+'/masks/'
        #self.edge_path = dataset_path +'/edgs/'

        self.edge_flage = hasEdg
        self.images = [self.img_path + f for f in os.listdir(self.img_path) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [self.mask_path + f for f in os.listdir(self.mask_path) if f.endswith('.png') or f.endswith(".jpg")]
       # self.edges = [self.edge_path + f for f in os.listdir(self.edge_path) if f.endswith('.png') or f.endswith(".jpg")]
        if self.augmentations :
            print('Using RandomRotation, RandomFlip')
            self.img_transform = transforms.Compose([
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(90, resample=False, expand=False, center=None),
                transforms.Resize(scale,Image.NEAREST),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            self.gt_transform = transforms.Compose([

                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(90, resample=False, expand=False, center=None),
                transforms.Resize(scale,Image.BILINEAR),
                transforms.ToTensor()])

        else:
            print('no augmentation')
            self.img_transform = transforms.Compose([
                transforms.Resize(scale,Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

            self.gt_transform = transforms.Compose([
                transforms.Resize(scale,Image.BILINEAR),
                transforms.ToTensor()])


    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
       # image = self.img_transform(image)
        #gt = self.gt_transform(gt)
        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        if self.img_transform is not None:
            image = self.img_transform(image)

        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)


        if self.edge_flage:
            edge = self.binary_loader(self.edges[index])
            random.seed(seed)  # apply this seed to img tranfsorms
            torch.manual_seed(seed)  # needed for torchvision 0.7
            edge = self.gt_transform(edge)
            return image, gt, edge
        else:
            return image, gt
       # return image, gt


    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')


    def __len__(self):
        return len(self.images)


class TestDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path,scale=(256,448)):
        super().__init__()
        self.img_path=dataset_path+'/images/'
        self.mask_path=dataset_path+'/masks/'
        self.images = [self.img_path + f for f in os.listdir(self.img_path) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [self.mask_path + f for f in os.listdir(self.mask_path) if f.endswith('.png') or f.endswith(".jpg")]
        self.img_transform = transforms.Compose([
            transforms.Resize((scale)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        # self.img_transform = transforms.Compose([
        #     transforms.Resize((scale)),
        #     transforms.ToTensor()])
        self.gt_transform = transforms.ToTensor()


    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        name = self.images[index].split('/')[-1]
        if name.endswith('.jpg'):
           name = name.split('.jpg')[0] + '_segmentation.png'
       # print(gt.shape[1:])
        return image,gt,name


    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')


    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
   data = Dataset('E:\dataset\dataset\TrainDataset')
   print(data.__getitem__(0))




