#!/bin/bash
temp=$1
list=/nfs/bigbox/hieule/penguin_data/p1000/$temp
root='/nfs/bigbox/hieule/penguin_data/p1000/PATCHES/64_386/'
#root='/nfs/bigbox/hieule/p1000/trainPATCHES/'
gpu=$2
dl=${3:-9998}
nc=3
bias=0.3
bs=128
name="MSEnc${nc}_${temp}_bias${bias}_bs${bs}"
checkpoint='/nfs/bigbox/hieule/penguin_data/checkpoints/'
CMD="python /nfs/bigbox/hieule/Penguins_CODE/penguin_train.py --batchSize $bs --biased_sampling $bias --traininglist $list --gpu_ids $gpu --model single_unet --input_nc $nc --name $name --root $root --dataroot ${root} --dataset_mode png_withlist --fineSize 256 --display_port ${dl} --checkpoints ${checkpoint}"
echo $CMD
eval $CMD
