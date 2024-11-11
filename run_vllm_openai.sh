#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

source activate hipporag
# nohup vllm serve meta-llama/Llama-3.1-70B-Instruct --dtype auto --port $2 --enable-prefix-caching --seed 0 --gpu-memory-utilization 0.95 --tensor-parallel-size 2 > /research/nfs_su_809/qi.658/vllm_log/vllm_output.log 2>&1 &
vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype auto --port $2 --enable-prefix-caching --seed 0 --gpu-memory-utilization 0.95 --tensor-parallel-size 1 --disable-log-requests
