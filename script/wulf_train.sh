#!/bin/bash
module load shared
module load cuda80/toolkit/8.0.44;nvcc INFILE -o OUTFILE;

#list=/gpfs/projects/LynchGroup/Penguin_workstation/Train_all/p1000/split2/all.txt

#root='/nfs/bigbox/hieule/p1000/trainPATCHES/'
gpu=$1
dl=${2:-9998}
nc=3
bias=$3
bs=128
fs=512
ls=768
PATCHES=512_768
name="bias${bias}_bs${bs}_P${PATCHES}_ls${ls}_fs${fs}_all"
root=/gpfs/projects/LynchGroup/Penguin_workstation/Train_all/fullsize/PATCHES/${PATCHES}/
checkpoint='/gpfs/projects/LynchGroup/Penguin_workstation/checkpoints'
CMD="python /gpfs/projects/LynchGroup/Penguin_workstation/Penguin_Code/Penguins/penguin_train.py --batchSize $bs --biased_sampling $bias --gpu_ids $gpu --model single_unet --input_nc $nc --name $name --root $root --dataroot ${root} --dataset_mode png --loadSize 768 --fineSize $fs --display_port ${dl} --checkpoints ${checkpoint} --display_host http://login"
echo $CMD
eval $CMD >$name.txt&

