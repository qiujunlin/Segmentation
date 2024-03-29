# coding:utf-8
import  os
import  torch
import  torch.nn as nn
import  torchvision.models as models
import  time
from config.config import DefaultConfig
import random
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
import logging
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


import  csv
def csv1():
    headers = ['dataset','net','train_path','recall', 'specificity', 'precision', 'F1', 'F2',
                           'ACC_overall', 'IoU_poly', 'IoU_bg', 'IoU_mean', 'Dice']
    with open('./data.csv', 'a', newline='')as f:
      f_csv = csv.writer(f)
      f_csv.writerow(headers)
     # rows=['1','2','3']
      #f_csv.writerow(rows)
    #f_csv.writerows(rows)
#@print(ner(1,2))
def rangess():
    for i in range(1,2):
        print(i)
import  tqdm
def charnge():
    args = DefaultConfig()

    save_path = os.path.join(args.data, args.dataset, "output")
    print(save_path)

def testtqdm():
    list =[]
    for i in range(1,100):
         list.append(i)
    pbar = tqdm.tqdm(enumerate(list, start=1), desc='Iter', total=100,
                     leave=False, bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:40}{r_bar}')
    for i  in  pbar:
     time.sleep(0.1)


def test2():
    pred =torch.randn(3,3)
    print(pred)
  # pred =torch.random.uniform(-1,1,size=[2,2])
    print((pred > 0).float().mean())
    print(torch.where(pred > 0))
    pred[torch.where(pred > 0)] /= (pred > 0).float().mean()
    pred[torch.where(pred < 0)] /= (pred < 0).float().mean()
    print(pred)
    pred = torch.sigmoid(pred).cpu().numpy() * 255
    print(pred)
def test3():
    res =  torch.rand(5,5)
    res= (res>0.5).float()
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    print(res)

def test4():
    logging.info("Dasd")
    logging.info("Dasd")
    logging.info("Dasd")


def  test5():
    for imagename in os.listdir("E:\dataset\dataset\TrainSmall\masks"):
        edge_map = binary2edge(os.path.join("E:\dataset\dataset\TrainSmall\masks",imagename))
        cv2.imwrite(os.path.join("E:\dataset\dataset\TrainSmall\edgs2", imagename), edge_map)


def test6():
    for imagename in os.listdir("F:\dataset\dataset\TrainDataset\masks"):

        mask = cv2.imread(os.path.join("F:\dataset\dataset\TrainDataset\masks",imagename), cv2.IMREAD_GRAYSCALE)
        edge_map =  mask_to_onehot(mask,2)
        edge_map = onehot_to_binary_edges(edge_map,2,2)
        cv2.imwrite(os.path.join("F:\dataset\dataset\TrainDataset\edgs", imagename), edge_map*255)

def mask_to_onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    _mask = [mask == i for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)
def onehot_to_binary_edges(mask, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to a binary edgemap (H,W)

    """
    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

    edgemap = np.zeros(mask.shape[1:])

    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :])+distance_transform_edt(1.0-mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edgemap += dist
    edgemap = np.expand_dims(edgemap, axis=0)
    edgemap = (edgemap > 0).astype(np.uint8)
    edgemap =  np.squeeze(edgemap)
    return edgemap
# def test6():
#     decay_rate = 0.1
#     decay_epoch =30
#     for epoch in range(1,100):
#       print(decay_rate ** (epoch // decay_epoch))


from dataset.Dataset import  TestDataset
from torch.utils.data import DataLoader
from torch.nn import functional as F
import utils.utils as u
import warnings
warnings.filterwarnings(action='ignore')
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
            pred1,pred2 = model(image)
            # eval Dice
            res = F.upsample(pred1+pred2 , size=gt.shape[2:], mode='bilinear', align_corners=False)
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

def test7():
    path ='F:\百度云下载\model_pth\PolypPVT.pth'
    from model import  PolypPVT
    from config import  config
    model = PolypPVT()
  #  model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(path))
    model.cuda()
    #model.cpu()
    model.eval()
    args =config.DefaultConfig()
    for dataset in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB','test']:
      dice =valid(model, dataset, args)
      print(dataset, ': ', dice)


def test9():
    path_ = r'E:\dataset\Ultro\TestDataset/test'
    path_true = r'E:\dataset\Ultro\TestDataset\test\masks'
    pathpred = r'E:\dataset\Ultro\TestDataset/test/output'
    TP = FPN = 0
    Jaccard = []
    for file in os.listdir(path_true):
        #            num=num+1
        pre_file_path = os.path.join(pathpred, file)
        true_file_path = os.path.join(path_true, file)
        img_true = np.array(Image.open(true_file_path).convert("L"))
        img_true=img_true/255
        img_pre = Image.open(pre_file_path).convert("L")
        img_pre == img_pre.resize(img_true.shape)
        img_pre = np.array(img_pre)
        img_pre = img_pre/255

       # img_pre = (img_true.shape)

        TP = TP + np.sum(img_pre * img_true)
        FPN = FPN + np.sum(img_pre) + np.sum(img_true)

    dice = 2 * TP / FPN

    print("DICE", dice)
def test10():
    seed = np.random.randint(2147483647)  # make a seed with numpy generator
    random.seed(seed)  # apply this seed to img tranfsorms
    torch.manual_seed(seed)  # needed for torchvision 0.7
    print(torch.rand(2,2))
    print(torch.rand(2,2))


    random.seed(seed)  # apply this seed to img tranfsorms
    torch.manual_seed(seed)  # needed for torchvision 0.7
    print(torch.rand(2, 2))
    print(torch.rand(2, 2))

import cv2
import  shutil
def test11():
    path ='F:\dataset\ISIC2018_Task1-2_Training_Input'
    dest=r"F:\dataset\isic2018\dataset\masks"
    for f in os.listdir(path):
        if 'superpixels' not   in f:
         sourcepath = os.path.join(path,f)
         img_array = cv2.imread(sourcepath, cv2.IMREAD_COLOR)
         new_name = os.path.join(dest,f)
         new_array = cv2.resize(img_array, (256, 192), interpolation=cv2.INTER_CUBIC)
         cv2.imwrite(new_name, new_array)
        # shutil.copy(sourcepath,new_name)
from PIL import Image

def test12():
    dest ='F:\dataset\isic2018\TrainDataset\masks'
    path =r"F:\dataset\ISIC2018_Task1_Training_GroundTruth\ISIC2018_Task1_Training_GroundTruth"
    for f in os.listdir(path):

        sourcepath = os.path.join(path,f)
        img_array = cv2.imread(sourcepath, cv2.IMREAD_COLOR)
        # 调用cv2.resize函数resize图片
        new_array = cv2.resize(img_array, (256,192), interpolation=cv2.INTER_CUBIC)
      #  f = f.replace(".jpg", ".png")
        newpath =  os.path.join(dest,f)

        cv2.imwrite(newpath, new_array)


def randchoice():

    pathimg =  "F:\dataset\isic2018\TrainDataset\images"
    pathmask =  "F:\dataset\isic2018\TrainDataset\masks"
    destimg=  "F:\dataset\isic2018\TestDataset/test\images"
    destmask=  "F:\dataset\isic2018\TestDataset/test/masks"

    imgs = []
    for x in os.listdir(pathimg):
        if x.endswith('png'):
            imgs.append(x)
    selected_imgs = random.sample(imgs, k=518)
    for img in selected_imgs :

        srcimg = os.path.join(pathimg, img)
        maskname =  img.replace(".png","")
        maskname =maskname+"_segmentation.png"
        srcmask = os.path.join(pathmask, maskname)
        dstim= os.path.join(destimg,   img)
        destm = os.path.join(destmask,   maskname)
        shutil.move(srcimg, dstim)
        shutil.move(srcmask, destm)


def test13():
    destimg=  "E:\dataset\dataset-video\dataset\TestDataset\CVC-ColonDB-300\images"
  #  destmask=  "E:\dataset\dataset-video\dataset\TestDataset\CVC-ClinicDB-612-Valid\images"
    destmask=  "E:\dataset\dataset-video\dataset\TestDataset/test\images"
    arr =[]
    arr2 =[]
    for f  in os.listdir(destimg) :
        arr.append(f)
    for f in os.listdir(destmask):
        if f  in arr :
            arr2.append(f)

    for  f in arr :
        if f not in arr2:
            print(f)

def test8():
    for f in os.listdir(r"E:\dataset\dataset-video\dataset\TestDataset\CVC-ColonDB-300\masks"):
        path = r"E:\dataset\dataset-video\dataset\TestDataset\CVC-ColonDB-300\masks\\" + f
        new_name = path.replace(".png", "-CVC-300.png")
        os.rename(path, new_name)

def test20():
    import cv2
    img = cv2.imread('E:\dataset\data\TestDataset\CVC-300\images/149.png')
    mask = cv2.imread('E:\dataset\data\TestDataset\CVC-300\masks/149.png')
    pred = cv2.imread('E:\dataset\data\TestDataset\CVC-300\output/149.png')

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(mask, 127, 255, 0)
    retpred, threshpred = cv2.threshold(pred, 127, 255, 0)
    contours, im = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 第一个参数是轮廓
    contourspred, im = cv2.findContours(threshpred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 第一个参数是轮廓
    cv2.drawContours(image=img, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=1)
    cv2.drawContours(image=img, contours=contourspred, contourIdx=-1, color=(255, 255, 0), thickness=1)

    cv2.namedWindow('a')
    cv2.imshow('a', img)
    cv2.waitKey(0)

def binary2edge(mask_path):
    """
    func1: threshold(src, thresh, maxval, type[, dst]) -> retval, dst
            https://www.cnblogs.com/FHC1994/p/9125570.html
    func2: Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]]) -> edges

    :param mask_path:
    :return:
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    ret, mask_binary = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)  # if <0, pixel=0 else >0, pixel=255
    mask_edge = cv2.Canny(mask_binary, 10, 150)

    return mask_edge


def binaryMask(im_path):
    im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
    ret, mask_binary = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY)

    return mask_binary
def test21():
    img = cv2.imread('E:\dataset\data\TestDataset\CVC-300\images/149.png')

# if __name__ == '__main__':
#     cv2.imshow("das",binary2edge(r"E:\dataset\Ultro\TrainDataset\masks/1-1.png"))
#     cv2.waitKey()
#     print(torch.__version__)
#     print(torch.cuda.is_available())

def distribution_map(mask, sigma):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 消除标注的问题孤立点

        dist1 = distance_transform_edt(mask)
        dist2 = distance_transform_edt(1 - mask)
        dist = dist1 + dist2
        dist = dist - 1

        f = lambda x, sigma: 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-x ** 2 / (2 * sigma ** 2))

        bdm = f(dist, sigma)

        bdm[bdm < 0] = 0

        return bdm * (sigma ** 2)

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



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1,relu=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Fusion(nn.Module):
    def __init__(self, channel, kernel_size=7):
        super(Fusion, self).__init__()
        self.relu = nn.ReLU(True)
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv2 = BasicConv2d(channel, channel, 3, padding=1)

        self.fuseconv = BasicConv2d(channel, channel, kernel_size=3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        self.conv_high = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv_low = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, high, low):
        low = self.conv1(low)
        high = self.conv2(high)

        avg_low = torch.mean(low, dim=1, keepdim=True)
        max_low, _ = torch.max(low, dim=1, keepdim=True)
        avg_high = torch.mean(high, dim=1, keepdim=True)
        max_high, _ = torch.max(high, dim=1, keepdim=True)

        avg_low_fu = avg_low * self.upsample(max_high)
        max_low_fu = max_low * self.upsample(avg_high)
        avg_high_fu = avg_high * self.downsample(max_low)
        max_high_fu = max_high * self.downsample(avg_low)

        low_fuse = self.conv_low(torch.cat((avg_low_fu, max_low_fu),dim=1))
        high_fuse = self.conv_high(torch.cat((avg_high_fu, max_high_fu), dim=1))

        low = self.sigmoid(low_fuse) * low
        high = self.sigmoid(high_fuse) * high

        fuse = self.fuseconv(low +self.upsample( high))

        return  fuse
def binary2edge(mask_path):
    """
    func1: threshold(src, thresh, maxval, type[, dst]) -> retval, dst
            https://www.cnblogs.com/FHC1994/p/9125570.html
    func2: Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]]) -> edges

    :param mask_path:
    :return:
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    ret, mask_binary = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)  # if <0, pixel=0 else >0, pixel=255
    mask_edge = cv2.Canny(mask_binary, 10, 150)

    return mask_edge
def  test22():
    #for imagename in os.listdir("E:\dataset\dataset\TrainDataset\masks"):
        imagename = "499.png"
        label_path = "E:\dataset\datasetnew\TrainDataset\masks/"+imagename
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
   #     label = np.float32(label > 128)
    #    label = distribution_map(label, 1)
     #   _edgemap = label.squeeze(axis=None)
      #  savepath = "E:\dataset\dataset\TrainDataset\edgs/"+imagename

       # cv2.imwrite(savepath, label *255)
        label =  binary2edge(label_path)
        label =  cv2.dilate(label,np.ones((5, 5), np.uint8),1)
        cv2.imshow("1",  label)
        cv2.waitKey(0)
def test8():
    for f in os.listdir(r"E:\dataset\BUSI\TrainDataset\masks"):
        path = r"E:\dataset\BUSI\TrainDataset\masks\\" + f
        new_name = path.replace("_mask.png", ".png")
        os.rename(path, new_name)

import numpy as np
import matplotlib.pyplot as plt

# 0 设置字体
plt.rc('font',family='Times New Roman', size=15)

# 1.1 定义sigmoid函数
def sigmoid(x):
    return 1. / (1 + np.exp(-x))
# 1.2 定义tanh函数
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
# 1.3 定义relu函数
def relu(x):
    return np.where(x < 0, 0, x)
# 1.4 定义prelu函数
def prelu(x):
    return np.where(x<0, x * 0.5, x)

# 2.1 定义绘制函数sigmoid函数
def plot_sigmoid(fig):
    x = np.arange(-10, 10, 0.1)
    y = sigmoid(x)
    ax = fig.add_subplot(2,2,1)#表示前面两个1表示1*1大小，最后面一个1表示第1个
    ax.spines['top'].set_color('none')#ax.spines设置坐标轴位置，set_color设置坐标轴边的颜色
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_position(('data', 0))
    ax.plot(x, y,color="black", lw=3)#设置曲线颜色，线宽
    plt.xlim([-10.05, 10.05])#设置坐标轴范围
    plt.ylim([-0.02, 1.02])
   # ax.set_title('(a) Sigmod')
   # plt.
  #  plt.savefig()
   # plt.show()#显示绘图
# 2.2 定义绘制函数tanh函数
def plot_tanh(fig):
    x = np.arange(-10, 10, 0.1)
    y = tanh(x)
    ax = fig.add_subplot(2,2,2)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.plot(x, y, color="black", lw=3)

    ax.set_yticks([-1.0, -0.5, 0.5, 1.0])
    ax.set_xticks([-10, -5, 5, 10])
    plt.xlim([-10.05, 10.05])#设置坐标轴范围
    plt.ylim([-0.02, 1.02])

  #  ax.set_title('(a) Tanh')
# 2.3 定义绘制函数relu函数
def plot_relu(fig):
    x = np.arange(-10, 10, 0.1)
    y = relu(x)
    ax = fig.add_subplot(2,2,3)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_position(('data', 0))
    ax.plot(x, y, color="black", lw=3)
    plt.xlim([-10.05, 10.05])#设置坐标轴范围
    plt.ylim([-0.02, 1.02])
    ax.set_yticks([2, 4, 6, 8, 10])
  #  ax.set_title('(c) ReLU')

# 2.4 定义绘制函数prelu函数
def plot_prelu(fig):
    x = np.arange(-10, 10, 0.1)
    y = prelu(x)

    ax = fig.add_subplot(2,2,4)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.plot(x, y, color="black", lw=3)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
  #  ax.set_title('(d) Leaky ReLU (a=0.5)')



def plt1():
    fig = plt.figure()
    plot_sigmoid(fig)
    plot_tanh(fig)
    plot_relu(fig)
    plot_prelu(fig)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.tight_layout()
    plt.show()




if __name__ == '__main__':
   plt1()

