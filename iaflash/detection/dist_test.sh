#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
CHECKPOINT=$2
GPUS=$3

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}


#./dist_test.sh  /workspace/mmdetection/configs/retinanet_r50_fpn_1x.py /model/retina/retinanet_r50_fpn_1x_20181125-7b0c2548 4
#./dist_test.sh  /workspace/mmdetection/configs/retinanet_r50_fpn_1x.py /model/retina/retinanet_r50_fpn_1x_20181125-7b0c2548 4
