#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
# conda activate flash

nohup python train.py experiment=disco_roberta_hatefuldiscussions2 logger=wandb env=rcohen_single hparams_search=disco_mdt &
