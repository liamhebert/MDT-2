#!/bin/bash

#SBATCH --time=8:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --partition=ALL
#SBATCH --mail-user=l2hebert@uwaterloo.ca
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=2
#SBATCH --signal=SIGUSR1@90

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun python train.py experiment=mdt_hatefuldiscussions2 trainer=ddp logger=wandb logger.wandb.project=mdt_hd2 env=all
