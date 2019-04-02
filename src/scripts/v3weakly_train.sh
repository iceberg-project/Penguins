#!/bin/bash

model=${1:-unetr}
gpu=$2
dl=${3:-9999}
NOTE=${4:-None}
root='/nfs/bigbox/hieule/GAN/data/Penguins/Train/PATCHES/128_384/'
wroot='/nfs/bigbox/hieule/GAN/data/Penguins/WL_Train/merged/PATCHES/192_384/'
#root='/nfs/bigbox/hieule/p1000/trainPATCHES/'
nc=3
bias=-1
bs=96
name="v3weakly_${model}_bs${bs}_${NOTE}"
TRAINING_POLICY="--save_epoch_freq 5 --niter 50 --niter_decay 950 --lr 0.0002"
checkpoint='/nfs/bigdisk/hieule/checkpoints_CVPR19W/'
CMD="python ../training/train.py  --randomSize --keep_ratio --batch_size $bs --biased_sampling $bias --gpu_ids $gpu --model $model --input_nc $nc --name $name --root $root --dataroot ${root} --dataset_mode v3weaklyanno --fineSize 256 --display_port ${dl} --checkpoints ${checkpoint} $TRAINING_POLICY --wdataroot $wroot"
echo $CMD
eval $CMD
