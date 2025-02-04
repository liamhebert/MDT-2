#!/bin/bash

nohup python train.py +trainer.detect_anomaly=true experiment=pretrain trainer=fsdp logger=wandb hydra=slurm_all hparams_search=default hydra.sweeper.n_jobs=1 &
