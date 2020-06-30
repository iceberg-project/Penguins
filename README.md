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
Login to Bridges via ssh using a Unix or Mac command line terminal.  Login is available to Bridges directly or through the XSEDE portal. Please see the <a href="https://portal.xsede.org/psc-bridges">Bridges User's Guide</a>.  

For Windows Users:  
Many tools are available for ssh access to Bridges.  Please see <a href="https://ubuntu.com/tutorials/tutorial-ubuntu-on-windows#1-overview">Ubuntu</a>, <a href="https://mobaxterm.mobatek.net/">MobaXterm</a>, or <a href="https://www.chiark.greenend.org.uk/~sgtatham/putty/">PuTTY</a>

### PSC Bridges
Once you have logged into Bridges, you can follow one of two methods for installing iceberg-penguins.

Method #1 (Recommended):  

(Note: The lines below starting with '$' are commands to type into your terminal.  Everything following '#' are comments to explain the reason for the command and should not be included in what you type.  Lines that do not start with '$' or '[penguins_env] $' are output you should expect to see.)

```bash
$ pwd
/home/username
$ cd $SCRATCH                      # switch to your working space.
$ mkdir Penguins                   # create a directory to work in.
$ cd Penguins                      # move into your working directory.
$ module load cuda                 # load parallel computing architecture.
$ module load python3              # load correct python version.
$ virtualenv penguins_env          # create a virtual environment to isolate your work from the default system.
$ source penguins_env/bin/activate # activate your environment. Notice the command line prompt changes to show your environment on the next line.
[penguins_env] $ pwd
/pylon5/group/username/Penguins
[penguins_env] $ export PYTHONPATH=<path>/penguins_env/lib/python3.5/site-packages # set a system variable to point python to your specific code. (Replace <path> with the results of pwd command above.
[penguins_env] $ pip install iceberg_penguins.search # pip is a python tool to extract the requested software (iceberg_penguins.search in this case) from a repository. (this may take several minutes).
```

Method #2 (Installing from source; recommended for developers only): 

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
[iceberg_penguins] $ deactivate    # exit your virtual environment.
$ interact -p GPU-small            # request a compute node (this may take a minute or two or more).
$ cd $SCRATCH/Penguins             # make sure you are in the same directory where everything was set up before.
$ module load cuda                 # load parallel computing architecture, as before.
$ module load python3              # load correct python version, as before.
$ source penguins_env/bin/activate # activate your environment, no need to create a new environment because the Penguins tools are installed and isolated here.
[iceberg_penguins] $ iceberg_penguins.detect --help  # this will display a help screen of available usage and parameters.
```

### Prediction
- Download a pre-trained model at: https://bit.ly/3eLSMuz

You download to your local machine and use scp, ftp, rsync, or Globus to transfer to bridges.

The one provided here is at the epoch 300 of the model we will call "MY_MODEL".

Please put the model file here: <checkpoints_dir>/MY_MODEL/

- The script to run the testing for a single PNG image tile:

iceberg_penguins.detect [--params ...]  
iceberg_penguins.detect --gpu-ids 0 --name MY_MODEL --epoch 300 --checkpoints_dir '../model_path/' --output test --input_im ../data/MY_IMG_TILE.png

## params:
- --gpu_ids: the gpu used for testing
- --name: name of the model used for testing
- --epoch: which epoch we use to test the model
- --checkpoints_dir: path to the folder containing the trained models
- --output: directory to save the outputs
- --input_im: path to the input image

