#!/bin/bash
gpu=$1
counter=1
dl=${2:-9996}
checkpoints_dir=/nfs/bigbox/hieule/penguin_data/checkpoints_new/
listfile=test_0.txt
list=/nfs/bigbox/hieule/penguin_data/p1000/split2/${listfile}
root='/nfs/bigbox/hieule/penguin_data/p1000/PATCHES/64_386/'
model=single_unet
checkpoint='/nfs/bigbox/hieule/penguin_data/checkpoints_new/'
bs=96
nc=3
epoch=25
counter=1
while [ $counter -le 4 ]
do
    echo $counter
    name=MSE_single_unet_train_1_${counter}.txt_bias-1_bs128_do0.1
    CMD="python /nfs/bigbox/hieule/Penguins_CODE/test_with_todolist.py --batchSize $bs --todolist $list --gpu_ids $gpu --model $model --input_nc $nc --name $name --root $root --dataroot ${root} --dataset_mode png_withlist --fineSize 256 --display_port ${dl} --checkpoints ${checkpoint} --which_epoch $epoch"
    echo $CMD
    eval $CMD
    ((counter++))
    echo $counter
done
