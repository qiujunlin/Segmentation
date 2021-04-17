import  os
import  torch
import numpy as np
import  torch.nn as nn
import  torchvision.models as models
from resnet import  resnet50
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
    net =  resnet50(pretrained=True)
    netdit  = net.state_dict()
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
u()