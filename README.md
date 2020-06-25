## Quality Metrics

[![Build Status](https://travis-ci.com/iceberg-project/Penguins.svg?branch=devel)](https://travis-ci.com/iceberg-project/Penguins)

## Prerequisites
- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Software Dependencies
- scipy==1.2.1
- Pillow==4.3.0
- torch
- scikit-learn==0.19.1
- torchvision==0.2.0'
- opencv-python
- rasterio
- future

## Installation

### PSC Bridges
From source:
```bash
$ git clone https://github.com/iceberg-project/Penguins.git
$ module load cuda
$ module load python3
$ virtualenv iceberg_penguins
$ source iceberg_penguins/bin/activate
[iceberg_penguins] $ export PYTHONPATH=<path>/iceberg_penguins/lib/python3.5/site-packages
[iceberg_penguins] $ pip install . --upgrade
```

From PyPi:
```bash
$ module load cuda
$ module load python3
$ virtualenv iceberg_penguins
$ source iceberg_penguins/bin/activate
[iceberg_penguins] $ export PYTHONPATH=<path>/iceberg_penguins/lib/python3.5/site-packages
[iceberg_penguins] $ pip install iceberg_penguins.search
```

To test
```bash
[iceberg_penguins] $ iceberg_penguins.detect
```

### Prediction
- Download a pre-trained model at:

https://drive.google.com/file/d/149j5rlynkO1jQTLOMpL5lextHY0ozw6N/view?usp=sharing

Please put the model file to: <checkpoints_dir>/<model_name>/

The one provided here is at the epoch 300 of the model named "v3weakly_unetr_bs96_main_model_ignore_bad"

- The script to run the testing for a single PNG image:

iceberg_penguins.detect [--params ...]

## params:
- --name: name of the model used for testing
- --gpu_ids: the gpu used for testing
- --checkpoints_dir: path to the folder containing the trained models
- --epoch: which epoch we use to test the model
- --input_im: path to the input image
- --output: directory to save the outputs

