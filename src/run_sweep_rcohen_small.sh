#!/bin/bash

nohup python train.py experiment=pretrain_bert_vit trainer=fsdp logger=wandb hydra=slurm hparams_search=default paths=scratch &
