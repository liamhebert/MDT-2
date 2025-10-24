#!/bin/bash

#SBATCH --time=16:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ALL
#SBATCH --mail-user=l2hebert@uwaterloo.ca
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=watgpu508
#SBATCH --signal=SIGUSR1@90

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

srun uv run train.py experiment=pretrain_v3_partial_to_full trainer=gpu logger=wandb logger.wandb.project=pretrain_to_full env=all
