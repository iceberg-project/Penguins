#!/bin/bash
module load shared
module load cuda80/toolkit/8.0.44;nvcc INFILE -o OUTFILE;

#list=/gpfs/projects/LynchGroup/Penguin_workstation/Train_all/p1000/split2/all.txt
root='/gpfs/projects/LynchGroup/Penguin_workstation/Train_all/fullsize/PATCHES/512_768/'

#root='/nfs/bigbox/hieule/p1000/trainPATCHES/'
gpu=0
dl=${2:-9998}
nc=3
bias=0.2
bs=128
name="MSEnc${nc}_${temp}_bias${bias}_bs${bs}_512_768_all"
checkpoint='/gpfs/projects/LynchGroup/Penguin_workstation/checkpoints'
CMD="python /gpfs/projects/LynchGroup/Penguin_workstation/Penguin_Code/Penguins/penguin_train.py --batchSize $bs --biased_sampling $bias --gpu_ids $gpu --model single_unet --input_nc $nc --name $name --root $root --dataroot ${root} --dataset_mode png --loadSize 768 --fineSize 256 --display_port ${dl} --checkpoints ${checkpoint} --display_host http://login"
echo $CMD
eval $CMD >$name.txt&

gpu=1
bias=-1
bs=128
dl=9999
name="MSEnc${nc}_${temp}_bias${bias}_bs${bs}_512_768_all"
checkpoint='/gpfs/projects/LynchGroup/Penguin_workstation/checkpoints'
CMD="python /gpfs/projects/LynchGroup/Penguin_workstation/Penguin_Code/Penguins/penguin_train.py --batchSize $bs --biased_sampling $bias --gpu_ids $gpu --model single_unet --input_nc $nc --name $name --root $root --dataroot ${root} --dataset_mode png --loadSize 768 --fineSize 256 --display_port ${dl} --checkpoints ${checkpoint} --display_host http://login"
echo $CMD
eval $CMD >$name.txt&

gpu=2
bias=0.5
bs=128
dl=9997
name="MSEnc${nc}_${temp}_bias${bias}_bs${bs}_512_768_all"
checkpoint='/gpfs/projects/LynchGroup/Penguin_workstation/checkpoints'
CMD="python /gpfs/projects/LynchGroup/Penguin_workstation/Penguin_Code/Penguins/penguin_train.py --batchSize $bs --biased_sampling $bias --gpu_ids $gpu --model single_unet --input_nc $nc --name $name --root $root --dataroot ${root} --dataset_mode png --loadSize 768 --fineSize 256 --display_port ${dl} --checkpoints ${checkpoint} --display_host http://login"
echo $CMD
eval $CMD >$name.txt&

gpu=3
bias=-1
bs=32
dl=9996
name="MSEnc${nc}_${temp}_bias${bias}_bs${bs}_512_768_all_fs512"
checkpoint='/gpfs/projects/LynchGroup/Penguin_workstation/checkpoints'
CMD="python /gpfs/projects/LynchGroup/Penguin_workstation/Penguin_Code/Penguins/penguin_train.py --batchSize $bs --biased_sampling $bias --gpu_ids $gpu --model single_unet --input_nc $nc --name $name --root $root --dataroot ${root} --dataset_mode png --loadSize 768 --fineSize 512 --display_port ${dl} --checkpoints ${checkpoint} --display_host http://login"
echo $CMD
eval $CMD >$name.txt
