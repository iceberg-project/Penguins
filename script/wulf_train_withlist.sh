#!/bin/bash
module load shared
module load cuda80/toolkit/8.0.44;nvcc INFILE -o OUTFILE;

list=/gpfs/projects/LynchGroup/Penguin_workstation/Train_all/p1000/split2/all.txt
root='/gpfs/projects/LynchGroup/Penguin_workstation/Train_all/p1000/PATCHES/64_386/'
#root='/nfs/bigbox/hieule/p1000/trainPATCHES/'
gpu=0
dl=${2:-9999}
nc=3
bias=-1
bs=128
name="MSEnc${nc}_${temp}_bias${bias}_bs${bs}"
checkpoint='/gpfs/projects/LynchGroup/Penguin_workstation/checkpoints'
CMD="python /gpfs/projects/LynchGroup/Penguin_workstation/Penguin_Code/Penguins/penguin_train.py --batchSize $bs --biased_sampling $bias --todolist $list --gpu_ids $gpu --model single_unet --input_nc $nc --name $name --root $root --dataroot ${root} --dataset_mode png_withlist --fineSize 256 --display_port ${dl} --checkpoints ${checkpoint} --display_host http://login"
echo $CMD
eval $CMD
