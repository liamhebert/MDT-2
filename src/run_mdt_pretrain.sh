#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --partition=ALL
#SBATCH --mail-user=l2hebert@uwaterloo.ca
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=2
#SBATCH --signal=SIGUSR1@90
#SBATCH --nodelist=watgpu408

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

nohup python train.py experiment=mdt_pretrain trainer=ddp logger=wandb logger.wandb.project=mdt_pretrain env=all hparams_search=seeds &
