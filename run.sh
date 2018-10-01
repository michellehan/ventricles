#!/bin/bash

if [ $# -eq 0 ]
then 
    script="param_train.py"
else
    script=$1
fi

args="$(python3.6 $script)"
CUDA_VISIBLE_DEVICES="0,1" python3.6 vent_main.py --parameters $script $args
