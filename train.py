# coding=gbk
import logging
import shutil
from torch.utils.data import DataLoader

import socket
from datetime import datetime
import os
import torch
from tensorboardX import SummaryWriter
import tqdm
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from  PIL import Image
import utils.utils as u

from config.config import DefaultConfig
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from dataset.Dataset import  TestDataset
import utils.loss as loss

"""
评价函数
"""

"""
导入模型 数据加载
"""

from dataset.Dataset import Dataset
from model.BaseNet import CPFNet
from model.resunet import  Resunet
from  model.mynet2 import MyNet
from  model.mynet3 import MyNet


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
    dice =u.AvgMeter()
    with torch.no_grad():
        for i, (data,gt,name) in enumerate(valid_dataloader):
            if torch.cuda.is_available() and args.use_gpu:
                img = data.cuda()
                gt = gt.cuda()

            out1, out2, out3, out4 = model(img)
            output = F.upsample(out4, size=gt.shape[2:], mode='bilinear', align_corners=False)
            output = torch.sigmoid(output)
            output = (output > 0.5).float()
            eps = 0.0001
            inter = torch.dot(output.view(-1), gt.view(-1))
            union = torch.sum(output) + torch.sum(gt) + eps
            t = (2 * inter.float() + eps) / union.float()
            dice.update(t)
    return  dice.avg






def train(args, model, optimizer,dataloader_train,total):
    Dicedict = {'CVC-300': [], 'CVC-ClinicDB': [], 'Kvasir': [], 'CVC-ColonDB': [], 'ETIS-LaribPolypDB': [],
                 'test': []}
    best_dice  =  0
    BCE = torch.nn.BCEWithLogitsLoss()
    for epoch in range(1, args.num_epochs+1):
        print("                            _ooOoo_                     ")
        print("                           o8888888o                    ")
        print("                           88  .  88                    ")
        print("                           (| -_- |)                    ")
        print("                            O\\ = /O                    ")
        print("                        ____/`---'\\____                ")
        print("                      .   ' \\| |// `.                  ")
        print("                       / \\||| : |||// \\               ")
        print("                     / _||||| -:- |||||- \\             ")
        print("                       | | \\\\\\ - /// | |             ")
        print("                     | \\_| ''\\---/'' | |              ")
        print("                      \\ .-\\__ `-` ___/-. /            ")
        print("                   ___`. .' /--.--\\ `. . __            ")
        print("                ."" '< `.___\\_<|>_/___.' >'"".         ")
        print("               | | : `- \\`.;`\\ _ /`;.`/ - ` : | |     ")
        print("                 \\ \\ `-. \\_ __\\ /__ _/ .-` / /      ")
        print("         ======`-.____`-.___\\_____/___.-`____.-'====== ")
        print("                            `=---='  ")
        print("                                                        ")
        u.adjust_lr(optimizer, args.lr, epoch, args.decay_rate, args.decay_epoch)
        size_rates = [0.75, 1, 1.25]  # replace your desired scale, try larger scale for better accuracy in small object
        model.train()
        loss_record = []
        loss_record1, loss_record2, loss_record3, loss_record4, loss_record5 = u.AvgMeter(), u.AvgMeter(), u.AvgMeter(), u.AvgMeter(), u.AvgMeter()
        for i, (data, label,edgs) in enumerate(dataloader_train, start=1):
            for rate in size_rates:


                #dataprepare
                if torch.cuda.is_available() and args.use_gpu:
                    data = data.cuda()
                    label = label.cuda()
                    edgs = edgs.cuda()

                 # rescale

                trainsize = int(round(args.trainsize * rate / 32) * 32)

                if   rate != 1:
                  data  = F.upsample(data, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                  label  = F.upsample(label, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                  edgs = F.upsample(edgs, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

                """
                网络训练 标准三步
                """
                optimizer.zero_grad()
                out1,out2 ,out3,out4= model(data)

                """
                计算损失函数
                """

                loss =  u.structure_loss(out1,label) + u.structure_loss(out2,label)+\
                        u.structure_loss(out3,label)+u.structure_loss(out4,label)
                loss.backward()

                u.clip_gradient(optimizer, args.clip)
                optimizer.step()

                loss_record.append(loss.item())

                # ---- train visualization ----
            if i % 20 == 0 or i == total:
                loss_train_mean = np.mean(loss_record)
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                          '[loss for train : {:.4f}]'.
                          format(datetime.now(), epoch, args.num_epochs, i, len(dataloader_train), loss_train_mean))

        if (epoch + 1) % 1 == 0:
            for dataset in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB']:
          # for dataset in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
                dataset_dice = valid(model, dataset,args)
                print("dataset:{},Dice:{:.4f}".format(dataset, dataset_dice))
                Dicedict[dataset].append(dataset_dice)
            meandice = valid(model, 'test',args )
            print("dataset:{},Dice:{:.4f}".format("test", meandice))
            Dicedict['test'].append(meandice)
            if meandice > best_dice:
                best_dice = meandice
                checkpoint_dir = "./checkpoint"
                filename = 'model_{}_{:03d}.pth.tar'.format(args.net_work, epoch)
                checkpointpath = os.path.join(checkpoint_dir, filename)
                torch.save(model.state_dict(), checkpointpath)

                print('#############  Saving   best  ##########################################BestAvgDice:{}'.format(best_dice))



def main():
    args = DefaultConfig()
    """
    create dataset and dataloader
    """

    dataset_train = Dataset(args.train_data_path, scale=(args.trainsize, args.trainsize))
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

    model_all={'MyNet':MyNet()}

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
        optimizer = torch.optim.AdamW(model.parameters(), args.lr,weight_decay=1e-4)
    else:
        optimizer =  torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    total = len(dataloader_train)

    train(args, model, optimizer,dataloader_train,total)


if __name__ == '__main__':
    # seed=1234
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)


    main()





