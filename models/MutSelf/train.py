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

"""
导入模型 数据加载
"""
from dataset.OCT import OCT
from model.BaseNet import CPFNet
from my_stacked_danet import DAF_stack
#from model.unet import  UNet
#from unet.unet_model import  U_Transformer
def train(args, model, optimizer,criterion, scheduler,dataloader_train, dataloader_val):
    #comments=os.getcwd().split('/')[-1]
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(args.log_dirs, current_time + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)
    step = 0
    best_pred=0.0

    softMax = nn.Softmax()
    CE_loss = nn.CrossEntropyLoss()
    mseLoss = nn.MSELoss()

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
#        is_best=False
        for i,(data, label) in enumerate(dataloader_train):
            # if i>9:
            #     break
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda().long()
            """
            网络训练 标准三步
            """
            optimizer.zero_grad()
          #  main_out = model(data)


            """
            计算损失函数
            
            """
            # semVector_1_1, \
            # semVector_2_1, \
            # semVector_1_2, \
            # semVector_2_2, \
            # semVector_1_3, \
            # semVector_2_3, \
            # semVector_1_4, \
            # semVector_2_4, \
            # inp_enc0, \
            # inp_enc1, \
            # inp_enc2, \
            # inp_enc3, \
            # inp_enc4, \
            # inp_enc5, \
            # inp_enc6, \
            # inp_enc7, \
            # out_enc0, \
            # out_enc1, \
            # out_enc2, \
            # out_enc3, \
            # out_enc4, \
            # out_enc5, \
            # out_enc6, \
            # out_enc7, \
            outputs0, \
            outputs1, \
            outputs2, \
            outputs3, \
            outputs0_2, \
            outputs1_2, \
            outputs2_2, \
            outputs3_2 = model(data)

            segmentation_prediction = (
                                              outputs0 + outputs1 + outputs2 + outputs3 + \
                                              outputs0_2 + outputs1_2 + outputs2_2 + outputs3_2
                                      ) / 8
            predClass_y = softMax(segmentation_prediction)


            # It needs the logits, not the softmax
            Segmentation_class = label

            # Cross-entropy loss
            loss0 = CE_loss(outputs0, Segmentation_class)
            loss1 = CE_loss(outputs1, Segmentation_class)
            loss2 = CE_loss(outputs2, Segmentation_class)
            loss3 = CE_loss(outputs3, Segmentation_class)
            loss0_2 = CE_loss(outputs0_2, Segmentation_class)
            loss1_2 = CE_loss(outputs1_2, Segmentation_class)
            loss2_2 = CE_loss(outputs2_2, Segmentation_class)
            loss3_2 = CE_loss(outputs3_2, Segmentation_class)
            #
            # lossSemantic1 = mseLoss(semVector_1_1, semVector_2_1)
            # lossSemantic2 = mseLoss(semVector_1_2, semVector_2_2)
            # lossSemantic3 = mseLoss(semVector_1_3, semVector_2_3)
            # lossSemantic4 = mseLoss(semVector_1_4, semVector_2_4)

            # lossRec0 = mseLoss(inp_enc0, out_enc0)
            # lossRec1 = mseLoss(inp_enc1, out_enc1)
            # lossRec2 = mseLoss(inp_enc2, out_enc2)
            # lossRec3 = mseLoss(inp_enc3, out_enc3)
            # lossRec4 = mseLoss(inp_enc4, out_enc4)
            # lossRec5 = mseLoss(inp_enc5, out_enc5)
            # lossRec6 = mseLoss(inp_enc6, out_enc6)
            # lossRec7 = mseLoss(inp_enc7, out_enc7)

            loss = (loss0 + loss1 + loss2 + loss3 + loss0_2 + loss1_2 + loss2_2 + loss3_2)
                    #+ 0.25 * (lossSemantic1 + lossSemantic2 + lossSemantic3 + lossSemantic4) \
                    #+ 0.1 * (
                     #           lossRec0 + lossRec1 + lossRec2 + lossRec3 + lossRec4 + lossRec5 + lossRec6 + lossRec7)  # CE_lossG
          #  loss_aux=criterion[0](main_out,label)
           # loss_main= criterion[1](main_out, label)
            #loss =loss_main+loss_aux
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
            Dice1,acc= u.val_softmax(args, model, dataloader_val)
            """
            更新学习率
            """
            if args.scheduler == 'CosineAnnealingLR':
                scheduler.step()
            elif args.scheduler == 'ReduceLROnPlateau':
                scheduler.step(Dice1)


            writer.add_scalar('Valid/Dice1_val', Dice1, epoch)
            writer.add_scalar('Valid/Acc_val', acc, epoch)

            """
            保存最好的dice,如果当前值比之前的大 就保存 否则就算了
            """

            is_best=Dice1 > best_pred
            best_pred = max(best_pred, Dice1)
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

            pred_RGB=OCT.COLOR_DICT[pred.astype(np.uint8)]
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
    dataset_train = OCT(dataset_path, scale=(args.crop_height, args.crop_width),mode='train')
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True 
    )
    
    dataset_val = OCT(dataset_path, scale=(args.crop_height, args.crop_width),mode='val')
    dataloader_val = DataLoader(
        dataset_val,
        # this has to be 1
        batch_size=len(args.cuda.split(',')),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True 
    )


    dataset_test = OCT(dataset_path, scale=(args.crop_height, args.crop_width),mode='test')
    dataloader_test = DataLoader(
        dataset_test,
        # this has to be 1
        batch_size=len(args.cuda.split(',')),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False 
    )

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    

    """
    load model
    """

    model_all={'BaseNet':CPFNet(out_planes=args.num_classes),
               'DAF_stack':DAF_stack()}
    model=model_all[args.net_work]
    print(args.net_work)
    cudnn.benchmark = True
    # model._initialize_weights()
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()
    # load pretrained model if exists
    if args.pretrained_model_path and mode=='test':
        print("=> loading pretrained model '{}'".format(args.pretrained_model_path))
        checkpoint = torch.load(args.pretrained_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        print('Done!')


    """
     optimizer and  scheduler 
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

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
    criterion_aux=nn.NLLLoss(weight=None)
    criterion_main=LS.Multi_DiceLoss(class_num=args.num_classes)
    criterion=[criterion_aux,criterion_main]




    if mode=='train':
        train(args, model, optimizer,criterion,scheduler,dataloader_train, dataloader_val)
    if mode=='test':
        test(model,dataloader_test, args)

if __name__ == '__main__':
    seed=1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    args=DefaultConfig()

    modes=args.mode

    if modes=='train':
        main(mode='train',args=args)
    elif modes=='test':
        main(mode='test',args=args)

