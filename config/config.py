# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:03:52 2019

@author: Administrator
"""
class DefaultConfig(object):
    num_epochs=100
    epoch_start_i=0
    checkpoint_step=5
    validation_step=1
    crop_height=256
    crop_width=448
    batch_size=1
    
    
    data='E:/dataset'
    dataset="data_med4"
    log_dirs='E:\workspace\python\CPFNet_Project\Log'
    
    lr=0.01#0.01
    lr_mode= 'poly'
    net_work= 'BaseNet'
    momentum = 0.9#
    weight_decay =1e-4#1e-4#


    mode='train'
    num_classes=2

    
    k_fold=4
    test_fold=4
    num_workers=1
    
    cuda='0'
    use_gpu=True
    pretrained_model_path='/root/PycharmProjects/CPFNet_Project/OCT/NET/checkpoints/model_RESUNT_012_0.7345.pth.tar'
    save_model_path='E:\workspace\python\CPFNet_Project\checkpoints'
    


