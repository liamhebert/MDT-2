#!/bin/sh

/opt/anaconda3/bin/python train.py experiment=pretrain_siglip logger=wandb env=all hparams_search=pretrain hydra/launcher=basic hydra.sweeper.n_jobs=1 hydra.sweeper.n_trials=1 dataset.dataset.debug=50 trainer=gpu trainer.max_epochs=1 trainer.min_epochs=1
