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

    def __init__(self, dataset_path="E:\dataset\dataset-video\dataset\VPS-TrainSet/",scale=(352,352),augmentations = True,video_dataset_list =["ASU-Mayo_Clinic", "CVC-ClinicDB-612", "CVC-ColonDB-300"]):
        super().__init__()
        self.augmentations = augmentations
        self.time_clips =5


        time_interval =1
        self.video_train_list = []
        for video_name in video_dataset_list:
            video_root = os.path.join(dataset_path, video_name, 'Train')
            cls_list = os.listdir(video_root)
            self.video_filelist = {}
            for cls in cls_list:
                self.video_filelist[cls] = []
                cls_path = os.path.join(video_root, cls)
                cls_img_path = os.path.join(cls_path, "Frame")
                cls_label_path = os.path.join(cls_path, "GT")
                tmp_list = os.listdir(cls_img_path)
                tmp_list.sort()
                for filename in tmp_list:
                    self.video_filelist[cls].append((
                        os.path.join(cls_img_path, filename),
                        os.path.join(cls_label_path, filename.replace(".jpg", ".png"))
                    ))
            # ensemble
            for cls in cls_list:
                li = self.video_filelist[cls]
                for begin in range(1, len(li) - (self.time_clips - 1) * time_interval - 1):
                #    batch_clips = []
                    for t in range(self.time_clips):
                       self.video_train_list.append(li[begin + time_interval * t])
                   # self.video_train_list.append(batch_clips)



        if self.augmentations :
            print('Using RandomRotation, RandomFlip')
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90, resample=False, expand=False, center=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize(scale,Image.NEAREST),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90, resample=False, expand=False, center=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
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
        image,gt = self.video_train_list[index]
        image = self.rgb_loader(image)
        gt = self.binary_loader(gt)
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
        return len(self.video_train_list)


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
        self.gt_transform = transforms.ToTensor()


    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        name = self.images[index].split('/')[-1]
        if name.endswith('.png'):
            name = name.split('.png')[0] + '.png'
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
   data = Dataset()
   print(data.__getitem__(0))
   print(data.__len__())




