#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup python train.py experiment=pretrain_siglip_gender_politics logger=wandb env=all hparams_search=pretrain &
