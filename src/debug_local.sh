#!/bin/sh

python train.py experiment=pretrain_siglip logger=wandb env=all hparams_search=pretrain hydra/launcher=basic hydra.sweeper.n_jobs=1 dataset.dataset.debug=100 trainer=fsdp
