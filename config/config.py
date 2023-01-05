# -*- coding: utf-8 -*-

class DefaultConfig(object):

    num_epochs=10
    epoch_start_i=0
    checkpoint_step=1
    validation_step=1
    crop_height= 352
    crop_width= 352
    trainsize = 64
    batch_size= 16


    train_data_path='/root/autodl-fs/datasetnew/TrainDataset'
    test_data_path='/root/autodl-fs/datasetnew/TestDataset'
    testdataset= ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']  #

    num_workers = 1

    clip =  0.5
    testsize =  (352,352)

    #optim
    optimizer = 'AdamW'
    lr=1e-4
    lr_mode= 'poly'
    net_work= 'MyNet'
    momentum = 0.9#
    weight_decay =1e-4#1e-4#
    decay_rate = 0.1
    decay_epoch =50
    cuda = '0'
    use_gpu = True

    num_classes=1
    augmentations =True



    pretrained_model_path='.\checkpoints\model_CBAMUnet_064_0.9335.pth.tar'
    save_model_path='E:\workspace\python\CPFNet_Project\checkpoints'
    


