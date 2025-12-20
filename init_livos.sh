#!/bin/bash

conda create -y -n livos python=3.11
conda activate livos
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install hydra-core==1.3.2
