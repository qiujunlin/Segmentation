# coding:utf-8
import  os
import  torch
import numpy as np
import  torch.nn as nn
import  torchvision.models as models
from PIL import Image
import  time
from config.config import DefaultConfig
from scipy import misc
import random
import cv2
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
import  numpy
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
def  test5():
    for imagename in os.listdir("E:\dataset\dataset\TrainSmall\masks"):
        edge_map = binary2edge(os.path.join("E:\dataset\dataset\TrainSmall\masks",imagename))
        cv2.imwrite(os.path.join("E:\dataset\dataset\TrainSmall\edgs2", imagename), edge_map)


def test6():
    for imagename in os.listdir("E:\dataset\dataset\TrainSmall\masks"):

        mask = cv2.imread(os.path.join("E:\dataset\dataset\TrainSmall\masks",imagename), cv2.IMREAD_GRAYSCALE)
        edge_map =  mask_to_onehot(mask,2)
        edge_map = onehot_to_binary_edges(edge_map,2,2)
        cv2.imwrite(os.path.join("E:\dataset\dataset\TrainSmall\edgs", imagename), edge_map*255)

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
def test6():
    decay_rate = 0.1
    decay_epoch =30
    for epoch in range(1,100):
      print(decay_rate ** (epoch // decay_epoch))
from dataset.Dataset import Dataset
from dataset.Dataset import  TestDataset
import shutil
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
    from model.pvt import  PolypPVT
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
def test8():

     for f in os.listdir(r"E:\dataset\Ultro\TestDataset/test\masks"):
             path = r"E:\dataset\Ultro\TestDataset/test\masks\\" + f
             new_name = path.replace("_mask", "")
             os.rename(path, new_name)


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



if __name__ == '__main__':

    print("ds")
    test10()
 # import numpy as np


