#!/bin/bash

#SBATCH --partition=low 
#SBATCH --gres=gpu:1      # Request GPU "generic resources"
#SBATCH --cpus-per-gpu=8  # Cores proportional to GPUs
#SBATCH --mem-per-gpu=32000M       # Memory proportional to GPUs
#SBATCH --output=out/%N-%j.out

#source $HOME/.poetry/env
#poetry shell


source $HOME/.poetry/env
poetry run python train.py fit --config configs/config.yaml