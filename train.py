# coding=gbk
from torch.utils.data import DataLoader
import warnings
# action参数可以设置为ignore，一位一次也不喜爱你是，once表示为只显示一次
warnings.filterwarnings(action='ignore')
import math
from datetime import datetime
import os
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn import functional as F
import numpy as np

import utils.utils as u
from torch import optim
from config.config import DefaultConfig
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
"""
评价函数
"""

"""
导入模型 数据加载
"""

from dataset.Dataset2 import Dataset
from dataset.Dataset2 import TestDataset

from model.idea2.MyNet4 import MyNet4
from   model.idea2.compare.model.BaseNet import CPFNet
from model.idea2.compare.Models.networks.network import Comprehensive_Atten_Unet
from model.idea2.compare.UNets import U_Net
from model.idea2.compare.UNets import AttU_Net
from model.idea2.compare.UNets import NestedUNet
from model.idea2.compare.core.res_unet_plus import ResUnetPlusPlus
from model.idea2.compare.core.res_unet import ResUnet


def valid(model, dataset,args):
    model.eval()
    data_path = os.path.join(args.test_data_path, dataset)
    dataset = TestDataset(data_path, args.testsize)
    valid_dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    avg =u.AvgMeter()
    with torch.no_grad():
        for i, (image,gt,name) in enumerate(valid_dataloader):
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            #a, b, c, d, e, f, g, h = model(image)
            pred = model(image)
            # eval Dice
            res = F.upsample(pred, size=gt.shape[2:], mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            input = res
            target = np.array(gt)
            smooth = 1
            input_flat = np.reshape(input, (-1))
            target_flat = np.reshape(target, (-1))
            intersection = (input_flat * target_flat)
            dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
            dice = '{:.4f}'.format(dice)
            dice = float(dice)
            avg.update(dice)
    return  avg.avg



def bdm_loss(pred, target, thresh=0.002, min_ratio=0.1):

    pred = pred.view(-1)
    target = target.view(-1)

    loss = F.mse_loss(pred, target, reduction='none')
    _, index = loss.sort()  # 从小到大排序

    threshold_index = index[-round(min_ratio * len(index))]  # 找到min_kept数量的hardexample的阈值

    if loss[threshold_index] < thresh:  # 为了保证参与loss的比例不少于min_ratio
        thresh = loss[threshold_index].item()

    loss[loss < thresh] = 0

    loss = loss.mean()

    return loss

def train(args, model, optimizer,dataloader_train,total):
    Dicedict = {'CVC-300': [], 'CVC-ClinicDB': [], 'Kvasir': [], 'CVC-ColonDB': [], 'ETIS-LaribPolypDB': [],
                 'test': []}
   # lr_lambda = lambda epoch: ((1 + math.cos(epoch * math.pi / args.num_epochs)) / 2) * (1 - 0.01) + 0.01
    lr_lambda = lambda epoch: 1.0 - pow((epoch / args.num_epochs), 0.9)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_dice=0
    best_epo =0
    BCE = torch.nn.BCEWithLogitsLoss()
    for epoch in range(1, args.num_epochs+1):
        size_rates = [ 0.75,1,1.25]  # replace your desired scale, try larger scale for better accuracy in small object
        model.train()
        loss_record = []
        for i, (data, label,edgs) in enumerate(dataloader_train, start=1):
            for rate in size_rates:

                #dataprepare
                if torch.cuda.is_available() and args.use_gpu:
                    data = Variable(data).cuda()
                    label = Variable(label).cuda()
                    edgs = Variable(edgs).cuda()

                    #  trainsize = int(round(args.trainsize * rate / 32) * 32)
                crop_height = int(round(args.crop_height * rate / 32) * 32)
                crop_width = int(round(args.crop_width * rate / 32) * 32)

                if rate != 1:
                    data = F.upsample(data, size=(crop_height, crop_width), mode='bilinear', align_corners=True)
                    label = F.upsample(label, size=(crop_height, crop_width), mode='bilinear', align_corners=True)
                    edgs = F.upsample(edgs, size=(crop_height, crop_width), mode='bilinear', align_corners=True)

                """
                网络训练 标准三步
                """
                optimizer.zero_grad()
              #  a, b, c, d, e,f,g,h =model(data)
                pred=  model(data)

                """
                计算损失函数
                """
               # lossb = bdm_loss(a,edgs) + bdm_loss(b,edgs) + bdm_loss(c,edgs) + bdm_loss(d,edgs)
                #lossg = u.structure_loss(e,label)+u.structure_loss(f,label)+u.structure_loss(g,label)+u.structure_loss(h,label)
                #loss = lossb +lossg
                loss = u.structure_loss(pred,label)
                loss.backward()

                u.clip_gradient(optimizer, args.clip)
                optimizer.step()

                loss_record.append(loss.item())

                # ---- train visualization ----
            if i % 20 == 0 or i == total:
                loss_train_mean = np.mean(loss_record)
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                          '[loss for train : {:.4f},lr:{:.7f}]'.
                          format(datetime.now(), epoch, args.num_epochs, i, len(dataloader_train), loss_train_mean, scheduler.get_last_lr()[0]))
        scheduler.step()
        if (epoch + 1) % 1 == 0:
            # for dataset in args.testdataset:
            #     dataset_dice = valid(model, dataset,args)
            #     print("dataset:{},Dice:{:.4f}".format(dataset, dataset_dice))
            #     Dicedict[dataset].append(dataset_dice)
            meandice = valid(model, 'test',args )
            print("dataset:{},Dice:{:.4f}".format("test", meandice))
            Dicedict['test'].append(meandice)
            if meandice > best_dice:
                best_dice = meandice
                best_epo =epoch
                checkpoint_dir = "/root/autodl-fs/checkpoints"
                filename = 'model_{}_{:03d}_{:.4f}.pth.tar'.format(args.net_work, epoch,best_dice)
                checkpointpath = os.path.join(checkpoint_dir, filename)
                if best_dice>0.7:
                  torch.save(model.state_dict(), checkpointpath)
                print('#############  Saving   best  ##########################################BestAvgDice:{}'.format(best_dice))
        print('bestepo:{:03d} ,bestdice :{:.4f}'.format(best_epo,best_dice))


def main():
    args = DefaultConfig()
    """
    create dataset and dataloader
    """

    dataset_train = Dataset(args.train_data_path, w=args.trainsize,h=args.trainsize,augmentations=args.augmentations,hasEdg=True)
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )


    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    """
    load model
    """

    model_all={'MyNet4':MyNet4(),'CPFNet':CPFNet(),'Comprehensive_Atten_Unet':Comprehensive_Atten_Unet(),
              'U_Net':U_Net(),'AttU_Net':AttU_Net(),'NestedUNet':NestedUNet(),'ResUnetPlusPlus':ResUnetPlusPlus(),'ResUnet':ResUnet()}

    model=model_all[args.net_work]
    print(args.net_work)
    cudnn.benchmark = True
    # model._initialize_weights()
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()



    """
     optimizer
    """
    if args.optimizer == 'AdamW':
        print("using AdamW")
        optimizer = torch.optim.AdamW(model.parameters(), args.lr,weight_decay=1e-4)
    else:
        print("using SGD")
        optimizer =  torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    total = len(dataloader_train)


    train(args, model, optimizer,dataloader_train,total)


if __name__ == '__main__':
    # seed=1234
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)


    main()





