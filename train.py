# coding=gbk
import logging
import shutil
from torch.utils.data import DataLoader
import warnings
# action参数可以设置为ignore，一位一次也不喜爱你是，once表示为只显示一次
warnings.filterwarnings(action='ignore')

from datetime import datetime
import os
import torch

from torch.nn import functional as F
import numpy as np

import utils.utils as u

from config.config import DefaultConfig
import torch.backends.cudnn as cudnn

from dataset.DatasetVideo import  TestDataset

from torch.autograd import Variable
"""
评价函数
"""

"""
导入模型 数据加载
"""

from dataset.Dataset import Dataset

from  model.mynet13_9 import MyNet



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
            prediction1, prediction2= model(image)
            # eval Dice
            res = F.upsample(prediction1+prediction2 , size=gt.shape[2:], mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            input = res
            target = np.array(gt)
            N = gt.shape
            smooth = 1
            input_flat = np.reshape(input, (-1))
            target_flat = np.reshape(target, (-1))
            intersection = (input_flat * target_flat)
            dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
            dice = '{:.4f}'.format(dice)
            dice = float(dice)
            avg.update(dice)
    return  avg.avg






def train(args, model, optimizer,dataloader_train,total):
    # Dicedict = {'CVC-300': [], 'CVC-ClinicDB': [], 'Kvasir': [], 'CVC-ColonDB': [], 'ETIS-LaribPolypDB': [],
    #              'test': []}
    Dicedict = {"CVC-ClinicDB-612-Test":[], "CVC-ClinicDB-612-Valid":[], "CVC-ColonDB-300":[],
                'test': []}
    best_dice=0
    best_epo =0
    BCE = torch.nn.BCEWithLogitsLoss()
    criterion = u.BceDiceLoss()
    for epoch in range(1, args.num_epochs+1):
        u.adjust_lr(optimizer, args.lr, epoch, args.decay_rate, args.decay_epoch)
        size_rates = [0.75, 1, 1.25]  # replace your desired scale, try larger scale for better accuracy in small object
        model.train()
        loss_record = []
        loss_record1, loss_record2, loss_record3, loss_record4, loss_record5 = u.AvgMeter(), u.AvgMeter(), u.AvgMeter(), u.AvgMeter(), u.AvgMeter()
        for i, (data, label) in enumerate(dataloader_train, start=1):
            for rate in size_rates:

                #dataprepare
                if torch.cuda.is_available() and args.use_gpu:
                    data = Variable(data).cuda()
                    label = Variable(label).cuda()
              #      edgs = Variable(edgs).cuda()

                 # rescale

                trainsize = int(round(args.trainsize * rate / 32) * 32)

                if   rate != 1:
                  data  = F.upsample(data, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                  label  = F.upsample(label, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
             #    edgs = F.upsample(edgs, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

                """
                网络训练 标准三步
                """
                optimizer.zero_grad()
                prediction1, prediction2 =model(data)

                """
                计算损失函数
                """

                loss = u.bce_dice(prediction1,label)+u.bce_dice(prediction2,label)
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
            for dataset in args.testdataset:
          # for dataset in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
                dataset_dice = valid(model, dataset,args)
                print("dataset:{},Dice:{:.4f}".format(dataset, dataset_dice))
                Dicedict[dataset].append(dataset_dice)
            meandice = valid(model, 'test',args )
            print("dataset:{},Dice:{:.4f}".format("test", meandice))
            Dicedict['test'].append(meandice)
            if meandice > best_dice:
                best_dice = meandice
                best_epo =epoch
                checkpoint_dir = "./checkpoint"
                filename = 'model_{}_{:03d}_{:.4f}.pth.tar'.format(args.net_work, epoch,best_dice)
                checkpointpath = os.path.join(checkpoint_dir, filename)
                torch.save(model.state_dict(), checkpointpath)

                print('#############  Saving   best  ##########################################BestAvgDice:{}'.format(best_dice))
        print('bestepo:{:03d} ,bestdice :{:.4f}'.format(best_epo,best_dice))


def main():
    args = DefaultConfig()
    """
    create dataset and dataloader
    """

    dataset_train = Dataset(args.train_data_path, scale=(args.trainsize, args.trainsize),augmentations=args.augmentations)
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





