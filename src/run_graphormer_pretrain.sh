#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --mem=48GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=ALL
#SBATCH --mail-user=l2hebert@uwaterloo.ca
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --signal=SIGUSR1@90

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

nohup python train.py experiment=graphormer_pretrain trainer=gpu logger=wandb logger.wandb.project=graphormer_pretrain env=all_single hparams_search=seeds &
