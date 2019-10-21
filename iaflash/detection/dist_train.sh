#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
GPUS=$2

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
    $(dirname "$0")/train.py $CONFIG --validate --resume_from 'work_dirs/retinanet_r50_fpn_1x/latest.pth' --launcher pytorch ${@:3}
