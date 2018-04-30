#!/bin/bash
nohup python -m visdom.server -p 9998 >&1 &
nohup python -m visdom.server -p 9999 >&1 &
#nohup ./script1.sh train10.5_ 1 9998 > 0.5.log >&1 &
nohup ./script1.sh train10.75_ 2 9999 > 0.75.log >&1 &
