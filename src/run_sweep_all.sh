#!/bin/bash

nohup python train.py experiment=pretrain trainer=fsdp logger=wandb hydra=slurm_all hparams_search=default &
