This is the implementation for ICCV submission **AutoFormer: Searching Transformers for Visual Recognition**.


## Environment Setup

To set up the enviroment you can easily run the following command:
```buildoutcfg
git clone https://github.com/ICCV2021/Autoformer.git
cd Autoformer
conda create -n Autoformer python=3.6
conda activate Autoformer
pip install -r requirements.txt
```

## Data Preparation 
You need to first download the [ImageNet-2012](http://www.image-net.org/) to the folder `./data/imagenet` and move the validation set to the subfolder `./data/imagenet/val`. To move the validation set, you cloud use the following script: <https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh>

The directory structure is the standard layout as following.
```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```


## Model Zoo
For evaluation, we provide the checkpoints of our models in [Google Drive](https://drive.google.com/drive/folders/17IHphEweUslPYgQ5CLAHyx6xnkuvAjqQ?usp=sharing).

After downloading the models, you can do the evaluation following the description in *Quick Start - Test*).

Model download links:

Model | FLOPs | Top-1 Acc. % | Top-5 Acc. % | Link 
--- |:---:|:---:|:---:|:---:
AutoFormer-T | 5.7M | 74.9 | 92.6 | [Google Drive](https://drive.google.com/file/d/1uRCW3doQHgn2H-LjyalYEZ4CvmnQtr6Q/view?usp=sharing) 
AutoFormer-S | 22.9M | 81.7 | 95.8 | [Google Drive](https://drive.google.com/file/d/1ldgVpN0ESksgctybuu3pHmdBcs7lByLf/view?usp=sharing) 
AutoFormer-B | 53.7M | 82.4 | 95.9 | [Google Drive](https://drive.google.com/file/d/1l2jiP3j9rc4O9rHi5RhyKk3l3X8pM-g6/view?usp=sharing)

## Quick Start
We provide *test* code of AutoFormer as follows.


### Test
To test our trained models, you need to put the downloaded model in `/PATH/TO/CHECKPOINT`. After that you could use the following command to test the model (Please change your config file/model checkpoint according to different models. Here we use the AutoFormer-B as an example).
```buildoutcfg
python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path /PATH/TO/IMAGENT --gp \
--change_qk --relative_position --mode retrain --dist-eval --cfg ./experiments/subnet/AutoFormer-B.yaml --resume /PATH/TO/CHECKPOINT --eval 
```



