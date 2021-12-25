import torch
import glob
import os
from torchvision import transforms
from PIL import Image
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia


class Dataset(torch.utils.data.Dataset):
    Unlabelled = [0, 0, 0]
    sick = [255, 255, 255]
    COLOR_DICT = np.array([Unlabelled, sick])

    def __init__(self, dataset_path, scale=(320, 320), mode='train'):
        super(Dataset, self).__init__()
        self.mode = mode
        self.img_path = dataset_path + '/img'
        self.mask_path = dataset_path + '/mask'
        self.image_lists, self.label_lists = self.read_list(self.img_path)
        self.resize = scale
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
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        # load image and crop
        img = Image.open(self.image_lists[index]).convert('RGB')
        img = img.resize(self.resize)
        img = np.array(img)
        labels = self.label_lists[index]
        # load label
        if self.mode == 'train':
            label_ori = Image.open(self.label_lists[index]).convert('RGB')
            label_ori = label_ori.resize(self.resize)
            label_ori = np.array(label_ori)
            label = np.ones(shape=(label_ori.shape[0], label_ori.shape[1]), dtype=np.uint8)

            # convert RGB  to one hot

            for i in range(len(self.COLOR_DICT)):
                equality = np.equal(label_ori, self.COLOR_DICT[i])
                class_map = np.all(equality, axis=-1)
                label[class_map] = i

            # augment image and label
            if self.mode == 'train':
                seq_det = self.flip.to_deterministic()  # 固定变换
                segmap = ia.SegmentationMapsOnImage(label, shape=label.shape)
                img = seq_det.augment_image(img)
                label = seq_det.augment_segmentation_maps([segmap])[0].get_arr().astype(np.uint8)

            label_img = torch.from_numpy(label.copy()).float()
            labels = label_img
        imgs = img.transpose(2, 0, 1) / 255.0
        img = torch.from_numpy(imgs.copy()).float()
        labels =labels.unsqueeze(0)
        return img, labels

    def __len__(self):
        return len(self.image_lists)

    def read_list(self, image_path):
        fold = os.listdir(image_path)
        img_list = []
        label_list = []
        if self.mode == 'train':
            for item in fold:
                name = item.split('.')[0]
                img_list.append(os.path.join(image_path, item))
                label_list.append(os.path.join(image_path.replace('img', 'mask'), '{}_mask.png'.format(name)))
        elif self.mode == 'val':
            for item in fold:
                name = item.split('.')[0]
                img_list.append(os.path.join(image_path, item))
                label_list.append(os.path.join(image_path.replace('img', 'mask'), '{}_mask.png'.format(name)))

        elif self.mode == 'test':
            for item in fold:
                name = item.split('.')[0]
                img_list.append(os.path.join(image_path, item))
                label_list.append(os.path.join(image_path.replace('img', 'mask'), '{}_mask.png'.format(name)))

        return img_list, label_list
