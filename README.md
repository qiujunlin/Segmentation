
实验结果
Start Validation!
Dice1: 0.783,ACC: 0.930: 100%|██████████| 86/86 [00:01<00:00, 66.11it/s]
Dice1: 0.7827096113788492
Acc: 0.9304280835528707

Start Validation!
Dice1: 0.783,ACC: 0.930: 100%|██████████| 86/86 [00:01<00:00, 66.11it/s]
Dice1: 0.7827096113788492
Acc: 0.9304280835528707


CBAM CosineAnnealingLR
Start Validation!
Dice1: 0.775,ACC: 0.931: 100%|██████████| 86/86 [00:01<00:00, 77.09it/s]
Dice1: 0.7753885218794664
Acc: 0.9308123921239099
最好 ：078


basenet  自定义lr
Start Validation!
Dice1: 0.794,ACC: 0.936: 100%|██████████| 86/86 [00:01<00:00, 64.61it/s]
Dice1: 0.7941000520211117
Acc: 0.9364795367979132
epoch 38, lr 0.008272: 100%|██████████| 344/344 [00:05<00:00, 60.14it/s, loss=0.113518]
:   0%|          | 0/86 [00:00<?, ?it/s]loss for train : 0.113518


attention unet 最好0 。78
Start Validation!
Dice1: 0.754,ACC: 0.926: 100%|██████████| 86/86 [00:02<00:00, 30.74it/s]
Dice1: 0.7541644626445958
Acc: 0.9258762499026681
epoch 199, lr 0.000085: 100%|██████████| 344/344 [00:27<00:00, 12.48it/s, loss=0.097189]
:   0%|          | 0/86 [00:00<?, ?it/s]loss for train : 0.097189



Start Validation!  daf 0.93
Dice1: 0.793,ACC: 0.120: 100%|██████████| 86/86 [00:02<00:00, 34.60it/s]
Dice1: 0.7928717929267244
Acc: 0.11985307357636005
epoch 199, lr 0.000085: 100%|██████████| 344/344 [00:20<00:00, 17.17it/s, loss=0.561584]
:   0%|          | 0/86 [00:00<?, ?it/s]loss for train : 0.561584

TP: 1433739.0
FPN: 3419565.0
DICE 0.8385505173903698
glob_Jaccard 0.7219862163150246
single_Jaccard 0.6589001024741271

Start Validation   unet++ 最号 0。78  有0。79   b=4
Dice1: 0.732,ACC: 0.922: 100%|██████████| 86/86 [00:01<00:00, 47.03it/s]
Dice1: 0.731515620164979
Acc: 0.9219108911051702
epoch 125, lr 0.004136: 100%|██████████| 344/344 [00:21<00:00, 16.33it/s, loss=10.436008]
:   0%|          | 0/86 [00:00<?, ?it/s]loss for train : 0.485396





## Prerequisites
- PyTorch 1.0   
   - `conda install torch torchvision`
- tqdm
   - `conda install tqdm`
- imgaug
   - `conda install six numpy scipy Pillow matplotlib scikit-image opencv-python imageio Shapely`
   - `conda install imgaug`
## 
