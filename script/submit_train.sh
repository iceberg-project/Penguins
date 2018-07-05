#!/bin/bash
module load torque/6.0.2
qsub -lwalltime=40:99:99 -q gpu-long wulf_train_fullsize.sh 
