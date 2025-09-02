#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# conda activate flash

nohup python train.py experiment=giga_pretrain_siglip_anchor logger=wandb env=rcohen hparams_search=pretrain &
