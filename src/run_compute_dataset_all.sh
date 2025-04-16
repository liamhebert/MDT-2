#!/bin/bash

#SBATCH --time=6:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=6
#SBATCH --partition=RCOHEN
#SBATCH --mail-user=l2hebert@uwaterloo.ca
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=1

python compute_dataset.py "$@" env=all hydra.launcher=local
