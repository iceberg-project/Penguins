#!/bin/bash
#nohup ./script1.sh train10.25_ 0 9998 > 0.25.log >&1 &
#nohup ./cv2_train_withlist.sh train1_all 1 9999 > train1_all.log >&1 &
nohup ./script2.sh 3 9996 > splittrain.log >&1 &
