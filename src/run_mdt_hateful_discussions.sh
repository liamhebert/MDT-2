#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rcohengpu:1
#SBATCH --partition=RCOHEN
#SBATCH --mail-user=l2hebert@uwaterloo.ca
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --signal=SIGUSR1@90

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train.py experiment=mdt_hatefuldiscussions2 trainer=ddp logger=wandb logger.wandb.project=mdt_hd2 env=all
