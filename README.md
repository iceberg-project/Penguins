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
- Download pre-trained model:
TBD

- The script to run the testing for a single PNG image:
python whole_image_prediction.py --gpu_ids 0





