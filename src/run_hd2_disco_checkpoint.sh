#!/bin/bash

#SBATCH --time=8:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=RCOHEN
#SBATCH --mail-user=l2hebert@uwaterloo.ca
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --signal=SIGUSR1@90

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
# conda activate flash
nohup python train.py experiment=disco_hatefuldiscussions2_ckpt_helpful_bird trainer=gpu logger=wandb env=all_single hparams_search=seeds &
