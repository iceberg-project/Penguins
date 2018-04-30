#!/bin/bash
GPU=$1
#module load cuda80/toolkit/8.0.44
#module load shared
#module load anaconda/3
#source activate pytorch
#root='/gpfs/projects/LynchGroup/CROZtrain/'
root='/nfs/bigbox/hieule/penguin_data/'
#code='/gpfs/home/hle/code/ADNET_demo/'
#trainingdir='CROPPED/p1000/training'
trainingdir='CROPPED/p300/PATCHES/64_386/'
name=single_p300_all
CMD="python ${code} penguin_train.py --gpu_ids $GPU --name $name --root $root --dataroot ${root}${trainingdir} --dataset_mode tif --fineSize 256 --display_port 9998 --checkpoints ${root}checkpoints"
echo $CMD
eval $CMD
#eval $CMD>/gpfs/home/hle/code/ADNET_demo/log/$name 
