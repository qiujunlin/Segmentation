from torch.utils.data import DataLoader
import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np

from model.idea2.MyNet4 import  MyNet4
from dataset.Dataset import  TestDataset
import  cv2

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=(352,352), help='testing size')
parser.add_argument('--pth_path', type=str, default='F:\checkpoint\model_MyNet4_154_0.8297.pth.tar')
if __name__ == '__main__':

 for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
 #or _data_name in ['test','val']:
    data_path = r'E:\dataset\dataset\TestDataset\{}\\'.format(_data_name)
    save_path = r'E:\dataset\dataset\TestDataset\{}\output/'.format(_data_name)
    edge_save_path = r'E:\dataset\dataset\TestDataset\{}\edgeoutput/'.format(_data_name)
   # edge_save_path = 'E:\dataset\dataset\TestDataset\{}\edgeoutput/'.format(_data_name)
    opt = parser.parse_args()
    model = MyNet4()
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
        a,b,c,d,e,f,g,h = model(img)
        pred = F.upsample(e, size=gt.shape[2:], mode='bilinear', align_corners=False)[0,0]
        edge_pred = F.upsample(a, size=gt.shape[2:], mode='bilinear', align_corners=False)[0,0]
        pred[torch.where(pred > 0)] /= (pred > 0).float().mean()
        pred[torch.where(pred < 0)] /= (pred < 0).float().mean()
        pred = torch.sigmoid(pred).cpu().detach().numpy() * 255
        edge_pred = torch.sigmoid(edge_pred).cpu().detach().numpy() * 255
       # res =(res>0.5)
     #   res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        #path =save_path+  "".join(name) 0.8995 0.9353 .9139 :0.7931 0.8135
        #path =save_path+  "".join(name) 0.895 0.930 .911 :0.789 0.807
        #path =save_path+  "".join(name) 0.895 0.931 .911 :0.791 0.807
        cv2.imwrite(save_path+name[0], np.round(pred))
        cv2.imwrite(edge_save_path+name[0], np.round(edge_pred))

