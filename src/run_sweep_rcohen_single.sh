#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup python train.py experiment=giga_pretrain_roberta_siglip_sup_con logger=wandb env=rcohen_single hparams_search=pretrain &
