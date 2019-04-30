#!/bin/bash

model=${1:-unet}
gpu=$2
dl=${3:-9999}
#root='/nfs/bigbox/hieule/GAN/data/Penguins/Train/PATCHES/128_384/'
root='/nfs/bigbox/hieule/GAN/data/Penguins/WL_Train/merged/PATCHES/192_384/'
nc=3
bias=0.5
bs=96
NOTE=${4:-None}
name="${model}_bs${bs}_sampling${bias}_${NOTE}"
TRAINING_POLICY="--save_epoch_freq 25 --niter 200 --niter_decay 800 --lr 0.002"
checkpoint='/nfs/bigdisk/hieule/checkpoints_CVPR19W/'
CMD="python ../training/train.py  --randomSize --keep_ratio --batch_size $bs --biased_sampling $bias --gpu_ids $gpu --model $model --input_nc $nc --name $name --root $root --dataroot ${root} --dataset_mode png --fineSize 256 --display_port ${dl} --checkpoints ${checkpoint} $TRAINING_POLICY"
echo $CMD
eval $CMD
