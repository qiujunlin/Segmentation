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

from model.BiDFNetS import  BiDFNet_s
from model.BiDFNet import  BiDFNet
from  model.BiDFNetOne import  BiDFNetOne
from model.BaseNet import  CPFNet
from dataset.Dataset import  TestDataset
from model.lib.TransFuse import   TransFuse_S
import  cv2
from model.Backbone import BackBone
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=(352,352), help='testing size')
parser.add_argument('--pth_path', type=str, default='F:\checkpoint\model_BiDFNet_012_0.8403.pth.tar')
# for _data_name in ['CVC-ClinicDB']:
#for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
if __name__ == '__main__':

 for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
 #for _data_name in ["CVC-ClinicDB-612-Test", "CVC-ClinicDB-612-Valid", "CVC-ColonDB-300"] :
 #or _data_name in ['test','val']:
    data_path = r'E:\dataset\data\TestDataset\{}\\'.format(_data_name)
    save_path = r'E:\dataset\data\TestDataset\{}\output/'.format(_data_name)
   # edge_save_path = 'E:\dataset\dataset\TestDataset\{}\edgeoutput/'.format(_data_name)
    opt = parser.parse_args()
    model = BiDFNet()
    model = torch.nn.DataParallel(model)
   # model.load_state_dict(torch.load(opt.pth_path)['state_dict'])
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()    
    #model.cpu()model_046_0.8890.pth.tar
    model.eval()

    os.makedirs(save_path, exist_ok=True)

   #os.makedirs(edge_save_path, exist_ok=True)
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
      #  img =  img.permute(0,2,3,1)
        img = img.cuda()
        prediction2,prediction1 = model(img)
        res = F.upsample(prediction1+prediction2, size=gt.shape[2:], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
       # res =(res>0.5)
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        #path =save_path+  "".join(name)
        cv2.imwrite(save_path+name[0], res*255)

        # edge = F.upsample(redfine1, size=gt.shape[2:], mode='bilinear', align_corners=False)
        # edge = edge.sigmoid().data.cpu().numpy().squeeze()
        # edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-8)
        #
        # cv2.imwrite(edge_save_path + name[0], edge * 255)
