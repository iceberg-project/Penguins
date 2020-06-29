## Quality Metrics

[![Build Status](https://travis-ci.com/iceberg-project/Penguins.svg?branch=devel)](https://travis-ci.com/iceberg-project/Penguins)

## Prerequisites - all available on bridges via the commands below
- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Software Dependencies - these will be installed automatically with the installation below.
- scipy==1.2.1
- Pillow==4.3.0
- torch
- scikit-learn==0.19.1
- torchvision==0.2.0'
- opencv-python
- rasterio
- future

## Installation
Preliminaries:
Login to bridges via ssh using a Unix or Mac command line terminal.  Login is available to bridges directly or through the XSEDE portal. Please see the [[Bridges User's Guide|https://portal.xsede.org/psc-bridges]]  
For Windows Users:  
Many tools are available for ssh access to bridges.  Please see [[Ubuntu|https://ubuntu.com/tutorials/tutorial-ubuntu-on-windows#1-overview]], [[MobaXterm|https://mobaxterm.mobatek.net/]], or [[PuTTY|https://www.chiark.greenend.org.uk/~sgtatham/putty/]]

### PSC Bridges
Recommended installation:  
When you are logged into bridges:  
From PyPi:  
Note: The lines below starting with '$' are commands to type into your terminal.  Everything following '#' are comments to explain the reason for the command and should not be included in what you type.  Lines that do not start with '$' or '[penguins_env] $' are output you should expect to see.
```bash
$ pwd
/home/username
$ cd $SCRATCH # switch to your working space
$ mkdir Penguins # create a directory to work in
$ cd Penguins # move into your working directory
$ module load cuda # load parallel computing architecture
$ module load python3 # load correct python version
$ virtualenv penguins_env # create a vitual environment to isolate your work from the default system
$ source penguins_env/bin/activate # activate your environment. Notice the command line prompt changes to show your environment on the next line.
[penguins_env] $ pwd
/pylon5/group/username/Penguins
[penguins_env] $ export PYTHONPATH=<path>/penguins_env/lib/python3.5/site-packages # set a system variable to point python to your specific code. (Replace <path> with the results of pwd command above.
[penguins_env] $ pip install iceberg_penguins.search # pip is a python tool to extract the requested software (iceberg_penguins.search in this case) from a repository.
```

Alternate installation (for developers):
From source:
```bash
$ git clone https://github.com/iceberg-project/Penguins.git
$ module load cuda
$ module load python3
$ virtualenv penguins_env
$ source penguins_env/bin/activate
[penguins_env] $ export PYTHONPATH=<path>/penguins_env/lib/python3.5/site-packages
[penguins_env] $ pip install . --upgrade
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
iceberg_penguins.detect --gpu-ids 0 --name v3weakly_unetr_bs96_main_model_ignore_bad --epoch 300 --checkpoints_dir '../checkpoints_CVPR19W/' --output test --input_im ../data/Penguins/Test/A/GE01_20120308222215_1050410000422100_12MAR08222215-M1BS-054072905140_01_P002_u08rf3031.png

## params:
- --gpu_ids: the gpu used for testing
- --name: name of the model used for testing
- --epoch: which epoch we use to test the model
- --checkpoints_dir: path to the folder containing the trained models
- --output: directory to save the outputs
- --input_im: path to the input image

