#!/bin/bash
clear
export PYTHONDONTWRITEBYTECODE=1

# env
source ~/anaconda3/etc/profile.d/conda.sh
conda activate torch

python nn.py
