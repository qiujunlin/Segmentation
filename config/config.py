# -*- coding: utf-8 -*-

class DefaultConfig(object):

    num_epochs=150
    epoch_start_i=0
    checkpoint_step=1
    validation_step=1
    crop_height=16
    crop_width=16
    trainsize = 352
    batch_size= 2


    #dataset
    train_data_path='E:\dataset\dataset\TrainSmall'
    test_data_path='E:\dataset\dataset\TestDataset'
    dataset="Kvasir"
    log_dirs='E:\workspace\python\CPFNet_Project\Log'
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
    decay_epoch =100
    cuda = '0'
    use_gpu = True

    num_classes=1
    augmentations =False



    pretrained_model_path='.\checkpoints\model_CBAMUnet_064_0.9335.pth.tar'
    save_model_path='E:\workspace\python\CPFNet_Project\checkpoints'
    


