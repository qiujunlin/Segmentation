# CPFNet_Projec
实验结果
Start Validation!
Dice1: 0.783,ACC: 0.930: 100%|██████████| 86/86 [00:01<00:00, 66.11it/s]
Dice1: 0.7827096113788492
Acc: 0.9304280835528707


## Folder
- `Dataset`: the folder where dataset is placed.
- `OCT`: the folder where model and model environment code are placed,`OCT` is the name of task. 
   - `dataset`: the file of data preprocessing.
   - `model`: model files.
   - `utils`: utils files(include many utils)
      - `config.py`: some configuration about project parameters.
      - `loss.py`: some custom loss functions
      - `utils.py`: some definitions of evaluation indicators
   - `metric.py`: offline evaluation function
   - `train.py`: training, validation and test function. 
- `Pretrain_model`:  pretriand encoder model,for example,resnet34.

## Prerequisites
- PyTorch 1.0   
   - `conda install torch torchvision`
- tqdm
   - `conda install tqdm`
- imgaug
   - `conda install six numpy scipy Pillow matplotlib scikit-image opencv-python imageio Shapely`
   - `conda install imgaug`
## 
