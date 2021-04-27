# coding:utf-8
import  os
import  torch
import numpy as np
import  torch.nn as nn
import  torchvision.models as models
from PIL import Image
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
def a():
    num = torch.rand((1,3,224,224))
    num =  torch.max(num,dim=1,keepdim=True)
    print(num[0].shape)
def test():
    net   =  SpatialAttention(kernel_size=3)
    ar   = torch.rand((1,3,224,224))
    print(net(ar))
    print(net(ar))
def avg():
    net = nn.AdaptiveAvgPool2d(1)
    num = torch.rand((1, 3, 224, 224))
    print(net(num).shape)
def model1():
    resnet = models.resnet50(pretrained=True)
    resnetdict  = resnet.state_dict()
    #net =  resnet18(pretrained=True)
   #netdit  = net.state_dict()
    # for k in netdit.keys():
    #     print(k)
    for k in resnetdict.keys():
        print(k)
def arr():
    a  = np.array([[0,0,0],[255,255,255]])
    b = torch.zeros((1,256,448)).data.cpu().numpy()
    c = a[b.astype(np.uint8)]
    print(c.shape)


def t():
    from torchvision import transforms
    scale =(256,448)
    resize_label = transforms.Resize(scale, Image.NEAREST)
    pre_file_path='E:\dataset\data_med4/test\predict/1-1_predict.png'
    img_pre = Image.open(pre_file_path).convert("L")
    img_pre = np.array(img_pre)
    img_pre[img_pre == 255] = 1
    true_file_path='E:\dataset\data_med4/test\mask/1-1_mask.png'
    img_true = Image.open(true_file_path).convert("L")
    img_true = resize_label(img_true)
    img_true = np.array(img_true)
    img_true[img_true == 255] = 1
    resize_img = transforms.Resize(scale, Image.BILINEAR)

    truefile  =   'E:\dataset\data_med4/test\img/1-1.png'
    true = Image.open(truefile)
    true =  resize_img(true)
    true =  np.array(true)
    true = true.transpose(2, 0, 1) / 255.0
    true=true[np.newaxis,:,:,:]
    true = torch.from_numpy(true.copy()).float()

    true = true.cuda()


    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import torch.backends.cudnn as cudnn
    pretrained_model_path = 'F:/checkpoints/2.tar'
    checkpoint = torch.load(pretrained_model_path)
    cudnn.benchmark = True
    from model.BaseNet import CPFNet
    model_all = {'BaseNet': CPFNet(out_planes=2)
                 }
    model = model_all['BaseNet']
    if torch.cuda.is_available():
         model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    true ,predicts= model(true)
    predict = torch.argmax(torch.exp(predicts), dim=1)
    pred_seg = predict.data.cpu().numpy()
    print(pred_seg.sum())
    print(img_true.sum())
    print(img_pre.sum())
def bceloss():
    a= torch.ones((1,1,2,2))
    b =torch.ones((1,1,2,2))
    bce_logit = nn.BCEWithLogitsLoss()
    losss  = bce_logit(a,b)
    print(losss)
def u():
    import  torch.nn.functional as F
    a =  torch.Tensor([0.98,0.1])
    c = torch.sigmoid(a)
    print(c)
def lr():
    print("1, lr 0.009910:")
    lr = 0.01
    for i in range(201):
      lr = lr * (1 - i / 200) ** 0.9
      print(lr)
def sum():
    a = torch.Tensor([[1,1],[1,1]])
    #print(a.sum(1))
    b=torch.rand((1,200))
    c=torch.rand((1,200))
    #print(c*b)
    print((c*b))
class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()

    def soft_dice_loss(self, y_true, y_pred):
        return 1
    def __call__(self, y_true, y_pred):
        a = 2
        return a
def roun():
     for classes in range(1,1):
         print("ada")

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention

def nolcal():
    net = Self_Attn(in_dim=8,activation=None)
    a =  torch.rand((1,8,112,112))
    print(net(a)[0].shape)
def ran1():
    for i in range(1,6):
        print(i)
"""
num_epochs=100
    epoch_start_i=0
    checkpoint_step=5
    validation_step=1
    crop_height=112
    crop_width=112
    batch_size=1
    #dataset
    data='E:/dataset/1050ti'
    dataset="data_med4"
    log_dirs='E:\workspace\python\CPFNet_Project\Log'
    k_fold = 4
    test_fold = 4
    num_workers = 1

    #optim
    lr=0.01#0.01  如果使用了scheduler 那么就设置为 0.001 如果使用的是不断下降 就使用 0.01
    lr_mode= 'poly'
    net_work= 'UNet'
    momentum = 0.9#
    weight_decay =1e-4#1e-4#

    # scheduler

    scheduler = ""  # 学习率优化器
    min_lr = 1e-5
    factor=0.1
    patience=2
    milestones='1,2'
    gamma=2/3
    early_stopping=-1

    # train and test way
    mode='train'
    num_classes=2

    # special model unet++
    deep_supervision = True

    
    cuda='0'
    use_gpu=True
    pretrained_model_path='E:\workspace\python\CPFNet_Project\checkpoints\model_BaseNet_053_0.8837.pth.tar'
    save_model_path='E:\workspace\python\CPFNet_Project\checkpoints'
"""

import  csv
def csv1():
    headers = ['net', 'train-epo', 'bestdice', 'bestepo', 'bestacc', 'lr', 'lr_mode', 'lastloss',
               'batchsize', 'crop_height', 'crop_width',
               'num_epochs', 'scheduler', 'momentum', 'weight_decay',
               'num_classes']
    with open('./data.csv', 'a', newline='')as f:
      f_csv = csv.writer(f)
      f_csv.writerow(headers)
      rows=['1','2','3']
      f_csv.writerow(rows)
    #f_csv.writerows(rows)
#@print(ner(1,2))
csv1()