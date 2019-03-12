#!/bin/bash

model=${1:-unet}
gpu=$2
dl=${3:-9999}
root='/mnt/train_ori/PATCHES/128_386/'
nc=3
bias=0.5
bs=96
name="${model}_bs${bs}_sampling${bias}"
checkpoint='/mnt/checkpoints_hackathon/'
CMD="python ../training/train.py  --randomSize --keep_ratio --batch_size $bs --biased_sampling $bias --gpu_ids $gpu --model $model --input_nc $nc --name $name --root $root --dataroot ${root} --dataset_mode png --fineSize 256 --display_port ${dl} --checkpoints ${checkpoint}"
echo $CMD
eval $CMD
