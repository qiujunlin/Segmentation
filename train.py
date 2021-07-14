# coding=gbk
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
import utils.loss as LS
from config.config import DefaultConfig
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from dataset.Dataset import  TestDataset
import  csv
"""
评价函数
"""
from utils.metrics import Metrics
from utils.metrics import evaluate
"""
导入模型 数据加载
"""

from dataset.Dataset import Dataset
from model.BaseNet import CPFNet
from model.resunet import  Resunet

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

def valid(model, valid_dataloader, total_batch):

    model.eval()

    # Metrics_logger initialization
    metrics = Metrics(['recall', 'specificity', 'precision', 'F1', 'F2',
                       'ACC_overall', 'IoU_poly', 'IoU_bg', 'IoU_mean','Dice'])

    with torch.no_grad():

        for i, (data,gt) in enumerate(valid_dataloader):

            if torch.cuda.is_available() and args.use_gpu:
                img = data.cuda()
                gt = gt.cuda()

            output = model(img)
            output = F.upsample(output, size=gt.shape[2:], mode='bilinear', align_corners=False)
            output = torch.sigmoid(output)
            _recall, _specificity, _precision, _F1, _F2, \
            _ACC_overall, _IoU_poly, _IoU_bg, _IoU_mean,_Dice= evaluate(output, gt, 0.5)

            metrics.update(recall= _recall, specificity= _specificity, precision= _precision,
                            F1= _F1, F2= _F2, ACC_overall= _ACC_overall, IoU_poly= _IoU_poly,
                            IoU_bg= _IoU_bg, IoU_mean= _IoU_mean,Dice=_Dice
                        )

    metrics_result = metrics.mean(total_batch)
    print('Dice:%f'%metrics_result["Dice"])
    return metrics_result



def train(args, model, optimizer,criterion, scheduler,dataloader_train, dataloader_val,total):


    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(args.log_dirs, current_time + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)
    step = 0
    best_pred=0.0


    for epoch in range(args.num_epochs):
        #动态调整学习率,使用官方的学习率调整和使用自己的
        if(args.scheduler==None):
         lr = optimizer.state_dict()['param_groups'][0]['lr']
        else:
         lr = u.adjust_learning_rate(args, optimizer, epoch)


        model.train()

        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []
        train_loss=0.0

        for i,(data, label) in enumerate(dataloader_train):
            # if i>9:
            #     break
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()
            """
            网络训练 标准三步
            """
            optimizer.zero_grad()
            main_out = model(data)

            """
            计算损失函数
            """
            # loss_aux=criterion[0](main_out,label)
            #loss_main= criterion[1](main_out, label)
            #loss =loss_main+loss_aux
            loss =  structure_loss(main_out,label)
            loss.backward()
            optimizer.step()


            tq.update(args.batch_size)
            train_loss += loss.item()

            tq.set_postfix(loss='%.6f' % (train_loss/(i+1))) #显示进度条信息
            step += 1

            if step%10==0:
                writer.add_scalar('Train/loss_step', loss, step)
            loss_record.append(loss.item())

        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('Train/loss_epoch', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))
        

        if epoch % args.validation_step == 0:
            metrics_result = valid(model, dataloader_val, total)
            Dice  =  metrics_result['Dice']
            """
            更新学习率
            """
            if   args.scheduler == 'CosineAnnealingLR':
                scheduler.step()
            elif args.scheduler == 'ReduceLROnPlateau':
                scheduler.step(metrics_result['Dice'])


            writer.add_scalar('Valid/Dice1_val', Dice, epoch)
            """
            保存最好的dice,如果当前值比之前的大 就保存 否则就算了
            """
            is_best=Dice > best_pred
            best_pred = max(best_pred, Dice)
            checkpoint_dir = args.save_model_path
            # checkpoint_dir=os.path.join(checkpoint_dir_root,str(k_fold))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            checkpoint_latest =os.path.join(checkpoint_dir, 'checkpoint_latest.pth.tar')
            u.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_dice': best_pred,
                   }, best_pred,epoch,is_best, args.net_work,checkpoint_dir,filename=checkpoint_latest)
    return best_pred


def test(model,dataloader, args):
    print('start test!')
    with torch.no_grad():
        model.eval()
        tq = tqdm.tqdm(dataloader,desc='\r')
        tq.set_description('test')

        comments=os.getcwd().split('/')[-1]

        for i, (data, label_path) in enumerate(tq):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                # label = label.cuda()
            """
            输出预测
            """
            aux_pred,predict = model(data)
            predict=torch.argmax(torch.exp(predict),dim=1)
            pred=predict.data.cpu().numpy()

            sum1 =  (pred==1).sum()

            pred_RGB=Dataset.COLOR_DICT[pred.astype(np.uint8)]
            sum2 = (pred_RGB[0,:,:,0]==255).sum()
            for index,item in enumerate(label_path):
                save_img_path=label_path[index].replace('mask','predict')
                if not os.path.exists(os.path.dirname(save_img_path)):
                    os.makedirs(os.path.dirname(save_img_path))
                img=Image.fromarray(pred_RGB[index].squeeze().astype(np.uint8))
                img.save(save_img_path)
                tq.set_postfix(str=str(save_img_path))
        tq.close()


def main(mode='train',args=None):


    """
    create dataset and dataloader
    """
    dataset_path = os.path.join(args.data, args.dataset)
    dataset_train = Dataset(dataset_path, scale=(args.crop_height, args.crop_width),mode='train')
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True 
    )
    
    dataset_val = TestDataset(dataset_path, scale=(args.crop_height, args.crop_width),mode='val')
    dataloader_val = DataLoader(
        dataset_val,
        # this has to be 1
        batch_size=len(args.cuda.split(',')),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True 
    )

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    

    """
    load model
    """

    model_all={'BaseNet':CPFNet(out_planes=args.num_classes),
               'Resunet':Resunet(out_planes=2)}

    model=model_all[args.net_work]
    print(args.net_work)
    cudnn.benchmark = True
    # model._initialize_weights()
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()
    # load pretrained model if exists



    """
     optimizer and  scheduler 
    """
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
    else:
        optimizer =  torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


    if args.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs, eta_min=args.min_lr)
    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience,
                                                   verbose=1, min_lr=args.min_lr)
    elif args.scheduler == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in args.milestones.split(',')], gamma=args.gamma)
    elif args.scheduler == 'ConstantLR':
        scheduler = None
    elif args.scheduler == "StepLR":
        scheduler = lr_scheduler.StepLR(optimizer,step_size=30)
    else:
        scheduler =None
    """
     loss
    """
    criterion_1=nn.BCEWithLogitsLoss(weight=None)
    criterion_2=LS.DiceLoss(class_num=args.num_classes)
    criterion=[criterion_1,criterion_2]


    if mode=='train':
        best_pred=train(args, model, optimizer,criterion,scheduler,dataloader_train, dataloader_val,dataset_val.__len__())
        return  best_pred

if __name__ == '__main__':
    seed=1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    args=DefaultConfig()

    modes=args.mode

    if modes=='train':
        bestdice = []
        for i in range(1,2):
          bestdice = []
          """
          训练
          """
          best_pred=main(mode='train',args=args)
          """
          保存参数
          """
          bestdice.append(best_pred)
          row=[args.net_work,i,best_pred,args.lr,args.lr_mode
               ,args.batch_size,args.crop_height,args.crop_width,args.num_epochs,args.scheduler,
               args.momentum,args.weight_decay,args.num_classes]
          """
          headers = [ 'net', 'train-epo','bestdice','bestepo', 'bestacc', 'lr', 'lr_mode','lastloss',
               'batchsize','crop_height','crop_width',
               'num_epochs','scheduler','momentum','weight_decay',
               'num_classes']
          """
          with open('./data.csv', 'a', newline='')as f:
              f_csv = csv.writer(f)
              f_csv.writerow(row)
        sum = 0
        for i, data in enumerate(bestdice):
            print("dice{}:{:.4f}".format(i, data))
            sum += data
        print("avgdice:{:.4f}".format(data / len(bestdice)))

    elif modes=='test':
        main(mode='test',args=args)

