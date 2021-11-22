import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import os, argparse
import imageio
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from scipy import misc

from model.mynet9 import  MyNet
from dataset.Dataset import  TestDataset
import  cv2

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=(352,352), help='testing size')
parser.add_argument('--pth_path', type=str, default='F:\checkpoint\model_MyNet_018_0.8214.pth.tar')
# for _data_name in ['CVC-ClinicDB']:
#for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
if __name__ == '__main__':

 for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
    data_path = 'E:\dataset\data\TestDataset\{}\\'.format(_data_name)

    save_path = 'E:\dataset\data\TestDataset\{}\output/'.format(_data_name)
    edge_save_path = 'E:\dataset\data\TestDataset\{}\edgeoutput/'.format(_data_name)
    opt = parser.parse_args()
    model = MyNet()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    #model.cpu()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(edge_save_path, exist_ok=True)
    test_loader1 = TestDataset(data_path, opt.testsize)
    test_loader = DataLoader(
        test_loader1,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=False
    )
    for i, (img,gt,name) in enumerate(test_loader):

        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        img = img.cuda()
        prediction1= model(img)
        res = F.upsample(prediction1, size=gt.shape[2:], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        #path =save_path+  "".join(name)
        cv2.imwrite(save_path+name[0], res*255)

        # edge = F.upsample(redfine1, size=gt.shape[2:], mode='bilinear', align_corners=False)
        # edge = edge.sigmoid().data.cpu().numpy().squeeze()
        # edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-8)
        #
        # cv2.imwrite(edge_save_path + name[0], edge * 255)
