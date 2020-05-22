#!/usr/bin/env bash

python eval_specialized_net.py \
    --net cpu_lat@17ms_top1@75.7_finetune@25 \
    --path /data/dataset/imagenet/ilsvrc2012/torchvision/

CUDA_VISIBLE_DEVICES=0 python eval_specialized_net.py \
    --net v100_gpu64@11ms_top1@76.1_finetune@25 \
    --path /data/dataset/imagenet/ilsvrc2012/torchvision/

