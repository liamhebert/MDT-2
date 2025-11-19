#!/bin/sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun python train.py experiment=giga_pretrain_roberta_anchor logger=wandb env=all hparams_search=pretrain hydra/launcher=basic hydra.sweeper.n_trials=1 trainer=gpu trainer.max_epochs=3 trainer.min_epochs=3 dataset.dataset.debug=40
