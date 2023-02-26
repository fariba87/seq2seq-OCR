#!/usr/bin/env bash
source /media/SSD1TB/rezaei/venvs/tf2.9/bin/activate
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.2/lib64
export CUDA_VISIBLE_DEVICES=1
python3.7 train.py --config  ./ConFig/GuidedCTC_cfg.json