#!/bin/bash

python train.py -d lyft \
    -r /home/xl3136/lyft_kitti \
    --p_pre 5 \
    --p_post 5 \
    --batch-size 6 \
    --num-workers 16 \
    --num-epoch 15