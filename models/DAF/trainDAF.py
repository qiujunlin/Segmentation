# coding=gbk
from torch.utils.data import DataLoader
from dataset.OCT import OCT
import socket
from datetime import datetime
import os
from model.unet import UNet
import torch
from tensorboardX import SummaryWriter
import tqdm
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import  transforms
import utils.utils as u
import utils.loss as LS
from config.config import DefaultConfig
import torch.backends.cudnn as cudnn
from models.resnext.model import  DAF


class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def eval_net(net, loader, device):
    print('\n')
    print('Start Validation!')
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)

            if net.n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_masks).item()
            pbar.update()

    net.train()
    return tot / n_val

def val(args, model, dataloader):
    print('\n')
    print('Start Validation!')
    with torch.no_grad():  # 在评价过程中停止求梯度  加快速度的作用
        model.eval()  # !!!评价函数必须使用
        tbar = tqdm.tqdm(dataloader, desc='\r')

        Acc = []
        cur_cube = []
        cur_label_cube = []
        next_cube = []
        counter = 0
        end_flag = False
        for i, (data, labels) in enumerate(tbar):
            # tbar.update()
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = labels[0].cuda()
            slice_num = labels[1].long().item()  # 获取总共的label数量  86
            # get RGB predict image

            predict = model(data)  # 预测结果  经过softmax后的 float32  1 1 n w
            #predict = torch.argmax(torch.exp(predicts), dim=1)  # int64 # n h w 获取的是结果 预测的结果是属于哪一类的
            #predict = torch.sigmoid(predict)
            batch_size = predict.size()[0]  # 当前的批量大小   1
            predict = (predict>0.5).float()
            counter += batch_size  # 每次加一
            if counter <= slice_num:  # 如果没有达到 总数
                cur_cube.append(predict)  # (1,h,w)
                cur_label_cube.append(label)  #
                if counter == slice_num:
                    end_flag = True
                    counter = 0
            else:  # 没用
                last = batch_size - (counter - slice_num)  # 6

                last_p = predict[0:last]
                last_l = label[0:last]

                first_p = predict[last:]
                first_l = label[last:]

                cur_cube.append(last_p)
                cur_label_cube.append(last_l)
                end_flag = True
                counter = counter - slice_num

            if end_flag:
                end_flag = False
                predict_cube = torch.stack(cur_cube, dim=0).squeeze()  # (n,h,w) int 64 tensor
                label_cube = torch.stack(cur_label_cube, dim=0).squeeze()  # n hw float32 tensor
                cur_cube = []
                cur_label_cube = []
                if counter != 0:  # w为0
                    cur_cube.append(first_p)
                    cur_label_cube.append(first_l)

                assert predict_cube.size()[0] == slice_num
                # 计算

                pred_seg = predict_cube.data.cpu().numpy()  # n h w int64 ndarray
                label_seg =label_cube.data.cpu().numpy().astype(dtype=np.int)  # n h w float32  -> int32 ndarray
                acc = (pred_seg * label_seg ).sum() / (
                            pred_seg.shape[0] * pred_seg.shape[1] * pred_seg.shape[2])  # acc 就是所有相同的像素值占总像素的大小

                overlap = (pred_seg * label_seg ).sum()
                union = (pred_seg).sum() + (label_seg).sum()
                dice=((2 * overlap + 0.1) / (union + 0.1))

                #Dice, true_label, acc = u.eval_multi_seg(predict_cube, label_cube, args.num_classes)

                # for class_id in range(args.num_classes - 1):
                #     if true_label[class_id] != 0:
                #         total_Dice[class_id].append(Dice[class_id])
                #Acc.append(acc)
                #len0 = len(total_Dice[0]) if len(total_Dice[0]) != 0 else 1
                # len1=len(total_Dice[1]) if len(total_Dice[1])!=0 else 1
                # len2=len(total_Dice[2]) if len(total_Dice[2])!=0 else 1

                #dice1 = sum(total_Dice[0]) / len0
                # dice2=sum(total_Dice[1])/len1
                # dice3=sum(total_Dice[2])/len2
                #ACC = sum(Acc) / len(Acc)
                # mean_dice=(dice1+dice2+dice3)/3.0
                tbar.set_description('Dice1: %.3f,ACC: %.3f' % (dice, acc))
        # print('Mean_Dice:',mean_dice)
        print('Dice1:', dice)
        # print('Dice2:',dice2)
        # print('Dice3:',dice3)
        print('Acc:', acc)

        return dice,acc


def train(args, model, optimizer, criterion, dataloader_train, dataloader_val):
    # comments=os.getcwd().split('/')[-1]
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(args.log_dirs, current_time + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)
    step = 0
    best_pred = 0.0
    for epoch in range(args.num_epochs):

        train_loss_record, loss0_record, loss1_record, loss2_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        loss3_record, loss0_2_record, loss1_2_record = AvgMeter(), AvgMeter(), AvgMeter()
        loss2_2_record, loss3_2_record = AvgMeter(), AvgMeter()

        lr = u.adjust_learning_rate(args, optimizer, epoch)
        model.train()
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []
        train_loss = 0.0
        #        is_best=False
        for i, (data, label) in enumerate(dataloader_train):
            # if i>9:
            #     break
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda().long()
            optimizer.zero_grad()

          #  aux_out, main_out = model(data)

            # get weight_map
            weight_map = torch.zeros(args.num_classes)
            weight_map = weight_map.cuda()
            for ind in range(args.num_classes):
                weight_map[ind] = 1 / (torch.sum((label == ind).float()) + 1.0)
            # print(weight_map)



          #  loss_aux = F.nll_loss(main_out, label, weight=None)
           # loss_main = criterion[1](main_out, label)

            #loss = loss_main + loss_aux

            #daf
            label = label.unsqueeze(1).float()
            outputs0, outputs1, outputs2, outputs3, outputs0_2, outputs1_2, outputs2_2, outputs3_2 = model(data)
            loss0 = criterion[2](outputs0, label)
            loss1 = criterion[2](outputs1, label)
            loss2 = criterion[2](outputs2, label)
            loss3 = criterion[2](outputs3, label)
            loss0_2 = criterion[2](outputs0_2, label)
            loss1_2 = criterion[2](outputs1_2, label)
            loss2_2 = criterion[2](outputs2_2, label)
            loss3_2 = criterion[2](outputs3_2, label)
            loss = loss0 + loss1 + loss2 + loss3 + loss0_2 + loss1_2 + loss2_2 + loss3_2

            loss.backward()
            optimizer.step()

            tq.update(args.batch_size)
            train_loss += loss.item()
            tq.set_postfix(loss='%.6f' % (train_loss / (i + 1)))  # 显示进度条信息
            step += 1
            if step % 10 == 0:
                writer.add_scalar('Train/loss_step', loss, step)
            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('Train/loss_epoch', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))

        if epoch % args.validation_step == 0:
            Dice1, acc = val(args, model, dataloader_val)

            writer.add_scalar('Valid/Dice1_val', Dice1, epoch)
            writer.add_scalar('Valid/Acc_val', acc, epoch)

            # mean_Dice=(Dice1+Dice2+Dice3)/3.0
            is_best = Dice1 > best_pred
            best_pred = max(best_pred, Dice1)
            checkpoint_dir = args.save_model_path
            # checkpoint_dir=os.path.join(checkpoint_dir_root,str(k_fold))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            checkpoint_latest = os.path.join(checkpoint_dir, 'checkpoint_latest.pth.tar')
            u.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_dice': best_pred,
            }, best_pred, epoch, is_best, args.net_work, checkpoint_dir, filename=checkpoint_latest)

to_pil = transforms.ToPILImage()
def test(model, dataloader, args):
    print('start test!')
    with torch.no_grad():
        model.eval()
        # precision_record = []
        tq = tqdm.tqdm(dataloader, desc='\r')
        tq.set_description('test')
        comments = os.getcwd().split('/')[-1]
        for i, (data, label_path) in enumerate(tq):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                # label = label.cuda()
            #predict = model(data)
            #predict = torch.argmax(torch.exp(predict), dim=1)

           # pred = predict.data.cpu().numpy()
           # sum1 = (pred == 1).sum()
            #pred_RGB = OCT.COLOR_DICT[pred.astype(np.uint8)]
            #sum2 = (pred_RGB[0, :, :, 0] == 255).sum()
            img_path =label_path[0].replace('mask','img')
            img_path =   img_path[:-8]+'.png'
            img = Image.open(img_path).convert("RGB")
            resize_img = transforms.Resize((256,448), Image.BILINEAR)
            img = resize_img(img)
            prediction = model(data)
            prediction=(prediction>0.5).float()
            prediction = np.array(to_pil(prediction.data.squeeze(0).cpu()))
            #prediction = np.array(to_pil(model(data).data.squeeze(0).cpu()))
           # prediction = misc.crf_refine(np.array(img), prediction)
            save_img_path = label_path[0].replace('mask', 'predict')
            Image.fromarray(prediction).save(save_img_path)
            # for index, item in enumerate(label_path):
            #     save_img_path = label_path[index].replace('mask', 'predict')
            #     if not os.path.exists(os.path.dirname(save_img_path)):
            #         os.makedirs(os.path.dirname(save_img_path))
            #     img = Image.fromarray(pred_RGB[index].squeeze().astype(np.uint8))
            #     img.save(save_img_path)
            #     tq.set_postfix(str=str(save_img_path))
        tq.close()


def main(mode='train', args=None):
    # create dataset and dataloader
    dataset_path = os.path.join(args.data, args.dataset)
    dataset_train = OCT(dataset_path, scale=(args.crop_height, args.crop_width), mode='train')
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    dataset_val = OCT(dataset_path, scale=(args.crop_height, args.crop_width), mode='val')
    dataloader_val = DataLoader(
        dataset_val,
        # this has to be 1
        batch_size=len(args.cuda.split(',')),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    dataset_test = OCT(dataset_path, scale=(args.crop_height, args.crop_width), mode='test')
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

    # load model
    model_all = {'DAF': DAF(),
                 'UNet': UNet(in_channels=3, n_classes=args.num_classes)}
    model = model_all[args.net_work]
    cudnn.benchmark = True
    # model._initialize_weights()
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()
    # load pretrained model if exists
    if args.pretrained_model_path and mode == 'test':
        print("=> loading pretrained model '{}'".format(args.pretrained_model_path))
        checkpoint = torch.load(args.pretrained_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        print('Done!')

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion_aux = nn.NLLLoss(weight=None)
    criterion_main = LS.Multi_DiceLoss(class_num=args.num_classes)
    bce_logit = nn.BCEWithLogitsLoss()
    criterion = [criterion_aux, criterion_main,bce_logit]
    if mode == 'train':
        train(args, model, optimizer, criterion, dataloader_train, dataloader_val)
    if mode == 'test':
        test(model, dataloader_test, args)

if __name__ == '__main__':
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    args = DefaultConfig()

    modes = args.mode

    if modes == 'train':
        main(mode='train', args=args)
    elif modes == 'test':
        main(mode='test', args=args)

