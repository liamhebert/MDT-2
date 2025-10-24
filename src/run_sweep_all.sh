#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1
# conda activate flash

echo "Launching pretrain v3 gemma contrastive sweep"
nohup uv run train.py experiment=pretrain_v3_gemma_contrastive logger=wandb env=all hparams_search=pretrain_v3 dataset.train_batch_size=10 dataset.test_batch_size=10 model.encoder.block_size=1 &
# echo "Launching pretrain v3 gemma supervised contrastive with ce sweep"
# nohup uv run train.py experiment=pretrain_v3_gemma_sup_con_ce logger=wandb env=all hparams_search=pretrain_v3 dataset.train_batch_size=10 dataset.test_batch_size=10 model.encoder.block_size=1 &
# echo "Launching pretrain v3 gemma supervised contrastive sweep"
# nohup uv run train.py experiment=pretrain_v3_gemma_sup_con logger=wandb env=all hparams_search=pretrain_v3 dataset.train_batch_size=10 dataset.test_batch_size=10 model.encoder.block_size=1 &
