#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup python train.py experiment=pretrain_siglip_single logger=wandb env=rcohen_single hparams_search=pretrain &
