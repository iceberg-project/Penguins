# U-NET in Pytorch for Image Segmentation
This repo is an implementation of U-Net for penguin colony detection. It is under active development.

This code is written by [Hieu-Le](https://lmhieu612.github.io). 

**Note**: The current software works well with PyTorch 0.4.



## Prerequisites
- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN



## Getting Started
### Installation
- Install PyTorch 0.4 and dependencies from http://pytorch.org
- Install Torch vision from the source.
```bash
git clone https://github.com/pytorch/vision
cd vision
python setup.py install
```
- Install python libraries [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate).
```bash
pip install visdom
pip install dominate
```
- Alternatively, all dependencies can be installed by
```bash
pip install -r requirements.txt
```
- Clone this repo:
```bash
git clone https://github.com/iceberg-project/Penguins/
```
### Prediction
- Download pre-trained model:
TBD

- The script to run the testing for a single PNG image:

python predict.py [--params ...]
params:
--name: name of the model used for testing
--gpu_ids: the ids of gpu
--checkpoints_dir: path to the folder containing the trained models
--epoch: which epoch we use to test the model
--input_im: path to the input image
--output: directory to save the outputs

Example script:
```bash
python predict.py --gpu-ids 0 --name model_name --epoch 300 --checkpoints_dir '/nfs/bigdisk/hieule/checkpoints_CVPR19W/' --output test --testset GE --input_im /nfs/bigbox/hieule/GAN/data/Penguins/Test/A/GE01_20120308222215_1050410000422100_12MAR08222215-M1BS-054072905140_01_P002_u08rf3031.png
```


