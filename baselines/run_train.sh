#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python train.py -d lyft \
    -r /home/xl3136/lyft_kitti \
    -m conv3d \
    --p_pre 5 \
    --p_post 5 \
    --batch-size 3 \
    --num-workers 8 \
    --num-epoch 15 \
    --amp