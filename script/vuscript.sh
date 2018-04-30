#!/bin/bash
gpu=$1
dl=${2:-9999}
root='/nfs/bigmind/vhnguyen/data/ivygap/train/'
nc=3
bs=64
name="VU"
bias=-0.5
checkpoint='/nfs/bigbox/hieule/VU_test/checkpoints/'
CMD="python penguin_train.py --randomSize --keep_ratio --batchSize $bs --biased_sampling $bias --gpu_ids $gpu --model vusingle_unet --input_nc $nc --name $name --root $root --dataroot ${root} --dataset_mode png --fineSize 256 --display_port ${dl} --checkpoints ${checkpoint}"
echo $CMD
eval $CMD
