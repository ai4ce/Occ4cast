#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python eval.py \
    -d results/lyft_p55_lr0.0005_batch6 \
    -r /home/xl3136/lyft_kitti