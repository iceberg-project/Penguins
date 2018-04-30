#!/bin/bash
temp=$1
file=${s//[[:blank:]]/}
cm=nohup $temp > ${file}.log 2>&1  </dev/null &
echo $cm
eval $cm
