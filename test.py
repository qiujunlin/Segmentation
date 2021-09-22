import torch
from tqdm import tqdm
from utils.metrics import evaluate
from torch.utils.data import DataLoader
from utils.metrics import Metrics
from dataset.Dataset import TestDataset
from config.config import DefaultConfig

import torch.nn.functional as F
import os
import  numpy as np
import imageio
from PIL import Image
from model.CBAMUnet import CBAMUnet


def generate_model(args):
    model_all = {'BaseNet': CBAMUnet(out_planes=args.num_classes)}
    model = model_all[args.net_work]
    model = torch.nn.DataParallel(model).cpu()
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()
    print("=> loading pretrained model '{}'".format(args.pretrained_model_path))
    model.load_state_dict(torch.load(args.pretrained_model_path, map_location=torch.device('cpu'))['state_dict'])
    return model


def test():
    print('loading test data......')
    args = DefaultConfig()
    model = generate_model(args)

    for dataset in tqdm.tqdm(args.testdataset, desc='Total TestSet', total=len(args.testdataset), position=0,
                         bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}'):

            dataset_path = os.path.join(args.data, args.dataset)
            dataset_test = TestDataset(dataset_path, scale=(args.crop_height, args.crop_width), mode='val')
            dataloader_test = DataLoader(
                dataset_test,
                batch_size=1,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False
            )
            save_path = os.path.join(args.data, args.dataset,"output")

            model.eval()

            with torch.no_grad():
                for i, (img, gt, name) in tqdm.tqdm(enumerate(dataloader_test), desc=dataset + ' - Test', total=len(dataloader_test), position=1, leave=False, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}'):
                    if torch.cuda.is_available() and args.use_gpu:
                        img = img.cuda()
                        gt = gt.cuda()

                    output = model(img)
                    out = F.upsample(output, size=gt.shape[2:], mode='bilinear', align_corners=False)
                    out = out.data.sigmoid().cpu().numpy().squeeze()
                    out = (out - out.min()) / (out.max() - out.min() + 1e-8)
                    Image.fromarray(((out > 0.5) * 255).astype(np.uint8)).save(os.path.join(save_path, name[0]))


if __name__ == '__main__':
    test()

    print('Done')
