#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
echo "CUDA environment configured: GPU=$CUDA_VISIBLE_DEVICES, TF_FORCE_GPU_ALLOW_GROWTH=$TF_FORCE_GPU_ALLOW_GROWTH"

python3 01_audio_features_extractor.py