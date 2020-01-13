#!/bin/bash

model=unetprior
gpu=$1
dl=${2:-9999}
NOTE=${5:-None}
root='/nfs/bigbox/hieule/GAN/data/Penguins/Train/PATCHES/128_384/'
wroot='/nfs/bigbox/hieule/GAN/data/Penguins/WL_Train/merged/PATCHES/192_384/'
#root='/nfs/bigbox/hieule/p1000/trainPATCHES/'
nc=3
bias=-1
bs=96
injectdepth=${3:-7}
priordepth=${4:-8}
priornf=12
wseg=500
wreg=100
wentr=0
wstrong=0.5
wposstrong=0.5
wposweak=0.5
datasetmode="priorweaklyanno"
name="v3weakly_${datasetmode}_${model}_bs${bs}_idepth${injectdepth}_pdepth${priordepth}_pnf${priornf}_seg${wseg}_reg${wreg}_entr${wentr}_S${wstrong}_${wposstrong}_${wposweak}_${NOTE}"
TRAINING_POLICY="--save_epoch_freq 20 --niter 100 --niter_decay 150 --lr 0.0001"
#checkpoint='/nfs/bigdisk/hieule/checkpoints_CVPR19W/'
checkpoint='/nfs/biglens/add_disk0/hieule/checkpoints_RS2020/'
CMD="python ../training/train.py  --randomSize --keep_ratio --batch_size $bs --biased_sampling $bias --gpu_ids $gpu --model $model --input_nc $nc --name $name --root $root --dataroot ${root} --dataset_mode ${datasetmode} --fineSize 256 --display_port ${dl} --checkpoints ${checkpoint} $TRAINING_POLICY --wdataroot $wroot --inject_depth $injectdepth --prior_depth $priordepth --prior_nf ${priornf} --lambda_segmentation ${wseg} --lambda_regression ${wreg} --lambda_entropy ${wentr} --s_strong $wstrong --s_pos_strong ${wposstrong} --s_pos_weak ${wposweak}"
echo $CMD
eval $CMD
