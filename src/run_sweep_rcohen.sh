#!/bin/bash

nohup python train.py experiment=pretrain_siglip logger=wandb hydra=slurm hparams_search=default paths=scratch &
