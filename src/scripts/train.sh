#!/bin/bash

model=${1:-unet}
gpu=$2
dl=${3:-9999}

root='/mnt/train_ori/PATCHES/128_386/'
#root='/nfs/bigbox/hieule/p1000/trainPATCHES/'
nc=3
bias=-1
bs=64
name="MSE_${model}_bs${bs}_MSE"
checkpoint='/mnt/checkpoints_hackathon/'
CMD="python ../training/train.py  --randomSize --keep_ratio --batchSize $bs --biased_sampling $bias --gpu_ids $gpu --model $model --input_nc $nc --name $name --root $root --dataroot ${root} --dataset_mode png --fineSize 256 --display_port ${dl} --checkpoints ${checkpoint}"
echo $CMD
eval $CMD
