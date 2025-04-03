#!/bin/sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train.py experiment=pretrain_siglip_gender_politics_many_token logger=wandb env=local hparams_search=pretrain hydra/launcher=basic hydra.sweeper.n_jobs=1 hydra.sweeper.n_trials=1 trainer=ddp trainer.max_epochs=3 trainer.min_epochs=3 dataset.dataset.debug=100
