import torch
from tqdm import tqdm
from utils.metrics import evaluate
from torch.utils.data import DataLoader
from utils.metrics import Metrics
from dataset.Dataset import  TestDataset
from config.config import DefaultConfig
import torch.nn.functional as F
import torch.nn.functional as F
import  os

from  model.BaseNet import  CPFNet


def generate_model(args):

    model_all = {'BaseNet': CPFNet(out_planes=args.num_classes)}
    model = model_all[args.net_work]
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()
    print("=> loading pretrained model '{}'".format(args.pretrained_model_path))
    model.load_state_dict(torch.load(args.pretrained_model_path)['state_dict'])
    return model

def test():
    print('loading data......')
    args =  DefaultConfig()
    dataset_path = os.path.join(args.data, args.dataset)
    dataset_test = TestDataset(dataset_path, scale=(args.crop_height, args.crop_width), mode='test')
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    total_batch = int(len(dataset_test) / 1)
    model = generate_model(args)
    model.cuda()

    model.eval()

    # metrics_logger initialization
    metrics = Metrics(['recall', 'specificity', 'precision', 'F1', 'F2',
                       'ACC_overall', 'IoU_poly', 'IoU_bg', 'IoU_mean', 'Dice'])

    with torch.no_grad():

        for i, (img,gt) in enumerate(dataloader_test):
            if torch.cuda.is_available() and args.use_gpu:
                img = img.cuda()
                gt = gt.cuda()

            output = model(img)

            output = F.upsample(output, size=gt.shape[2:], mode='bilinear', align_corners=False)
            output = torch.sigmoid(output)
            _recall, _specificity, _precision, _F1, _F2, \
            _ACC_overall, _IoU_poly, _IoU_bg, _IoU_mean, _Dice = evaluate(output, gt, 0.5)

            metrics.update(recall= _recall, specificity= _specificity, precision= _precision, 
                            F1= _F1, F2= _F2, ACC_overall= _ACC_overall, IoU_poly= _IoU_poly, 
                            IoU_bg= _IoU_bg, IoU_mean= _IoU_mean, Dice = _Dice
                        )

    metrics_result = metrics.mean(total_batch)

    print("Test Result:")
    print('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f, F2: %.4f, '
          'ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f, Dice:%.4f'
          % (metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'],
             metrics_result['F1'], metrics_result['F2'], metrics_result['ACC_overall'],
             metrics_result['IoU_poly'], metrics_result['IoU_bg'], metrics_result['IoU_mean'], metrics_result['Dice']))

if __name__ == '__main__':

    test()

    print('Done')
