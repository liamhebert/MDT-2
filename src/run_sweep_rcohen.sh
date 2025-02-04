#!/bin/bash

nohup python train.py experiment=pretrain trainer=fsdp logger=wandb hydra=slurm hparams_search=default &
