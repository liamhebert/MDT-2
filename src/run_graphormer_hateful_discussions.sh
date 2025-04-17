#!/bin/bash

#SBATCH --time=8:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ALL
#SBATCH --mail-user=l2hebert@uwaterloo.ca
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --signal=SIGUSR1@90

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun python train.py experiment=graphormer_hatefuldiscussions2 trainer=gpu logger=wandb logger.wandb.project=graphormer_hd2 env=all
