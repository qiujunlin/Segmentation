import torch
import os
import numpy as np
from torchvision import transforms
from PIL import Image
import random
import cv2
from scipy.ndimage.morphology import distance_transform_edt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms as T
class Dataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path,w=352,h=352,augmentations = True,hasEdg =False):
        super().__init__()
        self.augmentations = augmentations
        self.img_path=dataset_path+'/images/'
        self.mask_path=dataset_path+'/masks/'
        self.datapath =  dataset_path
        self.w=w
        self.h =h

        self.edge_flage = hasEdg
        self.images = [self.img_path + f for f in os.listdir(self.img_path) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [self.mask_path + f for f in os.listdir(self.mask_path) if f.endswith('.png') or f.endswith(".jpg")]

        self.samples = [name for name in os.listdir(self.datapath + '/images') if name[0] != "."]
        self.color1, self.color2 = [], []
        for name in self.samples:
            if name[:-4].isdigit():
                self.color1.append(name)
            else:
                self.color2.append(name)
        if self.augmentations == True:
            print("use  data augmentation !")
            self.transform = A.Compose([
                A.OneOf([
                    A.RandomResizedCrop(self.w, self.h, scale=(0.75, 1))
                ], p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
                A.OneOf([
                    A.CoarseDropout(max_holes=8, max_height=20, max_width=20, min_holes=None, min_height=None,
                                    min_width=None, fill_value=0, p=1),
                    A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=1),
                ], p=0.5),
               # A.Resize(352, 352),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5)
               # ToTensorV2()
            ])

        else:
            print("no data augmentation")
            self.transform = A.Compose([A.Resize(h,w)])
        self.as_tensor1 = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])
        self.as_tensor2 = T.Compose([
            T.ToTensor()
        ])


    def __getitem__(self, index):
        # image_path = self.images[index]
        # label_path = self.gts[index]
        # image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        # label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)# GRAY 1 channel ndarray with shape H * W
        #
        # seed = np.random.randint(2147483647)  # make a seed with numpy generator
        # random.seed(seed)  # apply this seed to img tranfsorms
        # torch.manual_seed(seed)  # needed for torchvision 0.
        # total = self.transform(image=image, mask=label)
        # image = total["image"]
        # label = total["mask"]
        name = self.samples[index]
        image = cv2.imread(self.datapath + '/images/' + name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        name2 = self.color1[index % len(self.color1)] if np.random.rand() < 0.7 else self.color2[index % len(self.color2)]
        image2 = cv2.imread(self.datapath + '/images/' + name2)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB)

        mean, std = image.mean(axis=(0, 1), keepdims=True), image.std(axis=(0, 1), keepdims=True)
        mean2, std2 = image2.mean(axis=(0, 1), keepdims=True), image2.std(axis=(0, 1), keepdims=True)
        image = np.uint8((image - mean) / std * std2 + mean2)
        image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
        mask = cv2.imread(self.datapath + '/masks/' + name, cv2.IMREAD_GRAYSCALE)
        mask = np.float32(mask > 128)
        pair = self.transform(image=image, mask=mask)


        image=  pair["image"]
        mask = pair["mask"]

        edge = self.distribution_map(mask, 0.5)
     #   cv2.imshow("1", edge * 255)
      #  cv2.waitKey(0)
     #   mask = mask/255.0
        return  self.as_tensor1(image),self.as_tensor2(mask),self.as_tensor2(edge)

    def distribution_map(self, mask, sigma):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 消除标注的问题孤立点

        dist1 = distance_transform_edt(mask)
        dist2 = distance_transform_edt(1 - mask)
        dist = dist1 + dist2
        dist = dist - 1

        f = lambda x, sigma: 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-x ** 2 / (2 * sigma ** 2))

        bdm = f(dist, sigma)

        bdm[bdm < 0] = 0

        return bdm * (sigma ** 2)

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
   data = Dataset('E:\dataset\dataset\TrainDataset',hasEdg=True)
   a,b,c= data.__getitem__(0)
   print(a)
   print(b)
   print(c)
   print(a.size())
   print(b.size())
   print(c.size())
   #print(a)




