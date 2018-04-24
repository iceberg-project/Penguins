#!/bin/bash
temp=$1
gpu=$2
counter=1
dl=${3:-9999}

while [ $counter -le 5 ]
do
    echo $counter
    list=/nfs/bigbox/hieule/penguin_data/p1000/${temp}${counter}
    root='/nfs/bigbox/hieule/penguin_data/p1000/PATCHES/64_386/'
    #root='/nfs/bigbox/hieule/p1000/trainPATCHES/'
    nc=3
    bias=0.2
    bs=128
    name="MSEnc${nc}_${temp}${counter}_bias${bias}_bs${bs}"
    checkpoint='/nfs/bigbox/hieule/penguin_data/checkpoints/'
    CMD="python penguin_train.py --batchSize $bs --biased_sampling $bias --traininglist $list --gpu_ids $gpu --model single_unet --input_nc $nc --name $name --root $root --dataroot ${root} --dataset_mode png_withlist --fineSize 256 --display_port ${dl} --checkpoints ${checkpoint}"
    echo $CMD
    eval $CMD
    ((counter++))
done
