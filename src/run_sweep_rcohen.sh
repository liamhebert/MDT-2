#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nohup python train.py experiment=pretrain_siglip_gender_politics_many_token logger=wandb env=rcohen hparams_search=pretrain &
