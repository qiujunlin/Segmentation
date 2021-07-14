# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:03:52 2019

@author: Administrator
"""
class DefaultConfig(object):
    num_epochs=30
    epoch_start_i=0
    checkpoint_step=5
    validation_step=1
    crop_height=352
    crop_width=352
    batch_size=1
    
    #dataset
    data='E:\dataset'
    dataset="Kvasir-SEG-900"
    log_dirs='E:\workspace\python\CPFNet_Project\Log'
    k_fold = 4
    test_fold = 4
    num_workers = 1


    #optim
    optimizer = 'Adam'
    lr=0.01#0.01  如果使用了scheduler 那么就设置为 0.001 如果使用的是不断下降 就使用 0.01
    lr_mode= 'poly'
    net_work= 'BaseNet'
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
    num_classes=1
    augmentations = 'False'

    # special model unet++
    deep_supervision = True

    
    cuda='0'
    use_gpu=True
    pretrained_model_path='E:\checkpoints\model_BaseNet_045_0.9215.pth.tar'
    save_model_path='E:\workspace\python\CPFNet_Project\checkpoints'
    


