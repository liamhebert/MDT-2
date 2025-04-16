#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

conda activate flash

nohup python train.py experiment=giga_pretrain_siglip logger=wandb env=rcohen hparams_search=pretrain &
