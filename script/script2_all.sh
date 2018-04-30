#!/bin/bash
gpu=$1
counter=0
dl=${2:-9996}

while [ $counter -le 5 ]
do
    echo $counter
    list=/nfs/bigbox/hieule/penguin_data/p1000/split/train_${counter}_all
    root='/nfs/bigbox/hieule/penguin_data/p1000/PATCHES/64_386/'
    #root='/nfs/bigbox/hieule/p1000/trainPATCHES/'
    nc=3
    bias=-1
    bs=128
    name="MSEnc${nc}_train_all_${counter}_bias${bias}_bs${bs}"
    checkpoint='/nfs/bigbox/hieule/penguin_data/checkpoints_new/'
    CMD="python /nfs/bigbox/hieule/Penguins_CODE/penguin_train.py --randomSize --keep_ratio --batchSize $bs --biased_sampling $bias --traininglist $list --gpu_ids $gpu --model single_unet --input_nc $nc --name $name --root $root --dataroot ${root} --dataset_mode png_withlist --fineSize 256 --display_port ${dl} --checkpoints ${checkpoint}"
    echo $CMD
    eval $CMD
    ((counter++))
    echo $counter
done
