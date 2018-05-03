#!/bin/bash
gpu=$2
l=$1
counter=0
dl=${3:-9996}

while [ $counter -le 4 ]
do
    echo $counter
    listfile=train_${l}_${counter}'.txt'
    list=/nfs/bigbox/hieule/penguin_data/p1000/split2/${listfile}
    root='/nfs/bigbox/hieule/penguin_data/p1000/PATCHES/64_386/'
    #root='/nfs/bigbox/hieule/p1000/trainPATCHES/'
    nc=3
    bias=-1
    bs=128
    dropout_w=0.1
    model=single_unet
    name="MSE_${model}_${listfile}_bias${bias}_bs${bs}_do${dropout_w}"
    checkpoint='/nfs/bigbox/hieule/penguin_data/checkpoints_new/'
    CMD="python /nfs/bigbox/hieule/Penguins_CODE/penguin_train.py --dropout_w $dropout_w --randomSize --keep_ratio --batchSize $bs --biased_sampling $bias --traininglist $list --gpu_ids $gpu --model $model --input_nc $nc --name $name --root $root --dataroot ${root} --dataset_mode png_withlist --fineSize 256 --display_port ${dl} --checkpoints ${checkpoint}"
    echo $CMD
    eval $CMD
    ((counter++))
    echo $counter
done
