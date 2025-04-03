#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rcohengpu:2
#SBATCH --partition=RCOHEN
#SBATCH --mail-user=l2hebert@uwaterloo.ca
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=2
#SBATCH --signal=SIGUSR1@90

srun python train.py "$@"
