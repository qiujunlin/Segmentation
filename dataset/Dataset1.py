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

    def __init__(self, dataset_path,w=352,h=352,augmentations = False,hasEdg =False):
        super().__init__()
        self.augmentations = augmentations
        self.img_path=dataset_path+'/images/'
        self.mask_path=dataset_path+'/masks/'
        self.w=w
        self.h =h

        self.edge_flage = hasEdg
        self.images = sorted([self.img_path + f for f in os.listdir(self.img_path) if f.endswith('.jpg') or f.endswith('.png')])
        self.gts = sorted([self.mask_path + f for f in os.listdir(self.mask_path) if f.endswith('.png') or f.endswith(".jpg")])
        if self.augmentations == True:
            print("use data augmentation!")
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
                A.Resize(352, 352)
            ])

        else:
            print("no data augmentation")
            self.transform = A.Compose([A.Resize(h,w)])
        self.as_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])
        self.as_tensor2 = T.Compose([
            T.ToTensor()
        ])


    def __getitem__(self, index):
        image_path = self.images[index]
        label_path = self.gts[index]
       # print(label_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W

        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.

        total = self.transform(image=image, mask=label)
        image = total["image"]
        mask = total['mask']
        if self.edge_flage:

            edge = self.binary2edge(mask)
         #   mask = mask / 255.0
          #  edge = edge / 255.0
            # cv2.imwrite("E:/1.png",mask)
           # cv2.imshow("1", edge)
            #cv2.waitKey(0)
            return self.as_tensor(image),self.as_tensor2(mask), self.as_tensor2(edge)
         #   return self.as_tensor(image),self.as_tensor2(mask), self.as_tensor2(edge)
        else:
            return self.as_tensor(image),torch.tensor(mask, dtype=torch.float)

    def binary2edge(self,mask):
        """
        func1: threshold(src, thresh, maxval, type[, dst]) -> retval, dst
                https://www.cnblogs.com/FHC1994/p/9125570.html
        func2: Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]]) -> edges

        :param mask_path:
        :return:
        """
        ret, mask_binary = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)  # if <0, pixel=0 else >0, pixel=255
        mask_edge = cv2.Canny(mask_binary, 10, 150)
        mask_edge = cv2.dilate(mask_edge, np.ones((2, 2), np.uint8), 1)
        return mask_edge

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
   data = Dataset('E:\dataset\datasetnew\TrainDataset',augmentations=True, hasEdg=True)
   a,b,c= data.__getitem__(0)
   print(a)
   print(b)
   print(c)
   print(a.size())
   print(b.size())
   print(c.size())
   #print(a)




