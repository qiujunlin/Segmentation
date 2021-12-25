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
from imgaug import augmenters as iaa
import imgaug as ia
class Dataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path,scale=(352,352),augmentations = True,hasEdg =False):
        super().__init__()
        self.augmentations = augmentations
        self.img_path=dataset_path+'/images/'
        self.mask_path=dataset_path+'/masks/'
        #self.edge_path = dataset_path +'/edgs/'
        self.scale =scale

        self.edge_flage = hasEdg
        self.images = [self.img_path + f for f in os.listdir(self.img_path) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [self.mask_path + f for f in os.listdir(self.mask_path) if f.endswith('.png') or f.endswith(".jpg")]
       # self.edges = [self.edge_path + f for f in os.listdir(self.edge_path) if f.endswith('.png') or f.endswith(".jpg")]
        self.flip = iaa.SomeOf((2, 5), [
            iaa.ElasticTransformation(alpha=(0, 50), sigma=(4.0, 6.0)),

            iaa.PiecewiseAffine(scale=(0, 0.1), nb_rows=4, nb_cols=4, cval=0),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.1),
            # iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
            iaa.Affine(rotate=(-10, 10),
                       scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
            iaa.OneOf([
                iaa.GaussianBlur((0, 1.0)),  # blur images with a sigma between 0 and 3.0
                iaa.AverageBlur(k=(3, 5)),  # blur image using local means with kernel sizes between 2 and 7
                iaa.MedianBlur(k=(3, 5)),  # blur image using local medians with kernel sizes between 2 and 7
            ]),
            iaa.contrast.LinearContrast((0.5, 1.5))],
                               random_order=True)


        self.img_transform = transforms.Compose([
                transforms.Resize(scale,Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

        self.gt_transform = transforms.Compose([
                transforms.Resize(scale,Image.BILINEAR),
                transforms.ToTensor()])


    def __getitem__(self, index):
        img = self.rgb_loader(self.images[index])
        label = self.binary_loader(self.gts[index])
        img = img.resize(self.scale)
        label =label.resize(self.scale)
        img = np.array(img)

        label =  np.array(label)
        label = label/255
        seq_det = self.flip.to_deterministic()  # 固定变换
        segmap = ia.SegmentationMapsOnImage(label, shape=label.shape)
        img = seq_det.augment_image(img)
        label = seq_det.augment_segmentation_maps([segmap])[0].get_arr().astype(np.uint8)

        imgs = img.transpose(2, 0, 1) / 255.0
        img = torch.from_numpy(imgs.copy()).float()
        label = torch.from_numpy(label.copy()).float()
        label = label.unsqueeze(0)


        return img,label



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
   data = Dataset(r'E:\dataset\Ultro\TrainDataset')
   print(data.__getitem__(0))




