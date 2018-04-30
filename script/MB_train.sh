#!/bin/bash
temp=p500_train
#root='/nfs/bigbox/hieule/penguin_data/CROPPED/'${temp}'/PATCHES/64_386/'
root='/nfs/bigbox/hieule/penguin_data/MB_Same_Size/Train/Train_all/CROPPED/'${temp}'/PATCHES/64_386'
#root='/nfs/bigbox/hieule/p1000/trainPATCHES/'
gpu=$1
dl=9999
nc=4
bias=-0.5
bs=128
datasetmode=mb
name="MSE_${datasetmode}_${temp}_bias${bias}_bs${bs}"
checkpoint='/nfs/bigbox/hieule/penguin_data/checkpoints/'
CMD="python penguin_train.py --batchSize $bs --biased_sampling $bias  --gpu_ids $gpu --model single_unet_mb --input_nc $nc --name $name --root $root --dataroot ${root} --dataset_mode mb --fineSize 256 --display_port ${dl} --checkpoints ${checkpoint}"
echo $CMD
eval $CMD
