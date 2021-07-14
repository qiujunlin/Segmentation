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
from model.BaseNet import CPFNet

from dataset.Dataset import Dataset
from model.BaseNet import CPFNet
from model.CBAMUnet import  CBAMUnet
from model.resunet import  Resunet
from dataset.Dataset import  TestDataset


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=(352,352), help='testing size')
parser.add_argument('--pth_path', type=str, default='E:\checkpoints\model_CBAMUnet_064_0.9335.pth.tar')
# for _data_name in ['CVC-ClinicDB']:
#for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
if __name__ == '__main__':

 for _data_name in [ 'Kvasir-SEG-900']:

    ##### put ur data_path here #####
    data_path = 'E:\dataset/{}/'.format(_data_name)
    #####                       #####

    save_path = 'E:\dataset\{}\output/'.format(_data_name)
    opt = parser.parse_args()
    model = CBAMUnet()
    model = torch.nn.DataParallel(model).cpu()
    model.load_state_dict(torch.load(opt.pth_path,map_location=torch.device('cpu'))['state_dict'])
 #   model.cuda()
    #model.cpu()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    test_loader1 = TestDataset(data_path, opt.testsize,mode='val')
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
        #image = image.cuda()

        res = model(img)
        res = F.upsample(res, size=gt.shape[2:], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        path =save_path+  "".join(name)
        imageio.imwrite(path, res)