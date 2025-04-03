#!/bin/bash

#SBATCH --time=8:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=6
#SBATCH --partition=RCOHEN
#SBATCH --mail-user=l2hebert@uwaterloo.ca
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --nodelist=watgpu102

python compute_dataset.py "$@" env=rcohen hydra.launcher=local
