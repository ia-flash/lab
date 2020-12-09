#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

SCRIPT=$1
CONFIG=$2
CHECKPOINT=$3
GPUS=$4

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
    $(dirname "$0")/$SCRIPT $CONFIG $CHECKPOINT  --launcher pytorch ${@:5}


#./dist_test.sh  test.py /workspace/mmdetection/configs/retinanet/retinanet_r50_fpn_1x_coco.py /model/retina/retinanet_r50_fpn_1x_20181125-7b0c2548.pth 4
#./dist_test.sh test_person.py /model/person_detection/retinanet_r50_fpn_1x.py  /model/person_detection/retinanet_r50_fpn_1x/latest.pth 4
# python /mmdetection/mmdet/utils/collect_env.py
