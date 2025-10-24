#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# conda activate flash

nohup uv run train.py experiment=pretrain_v3_gemma_sup_con logger=wandb env=rcohen hparams_search=pretrain_v3 &
