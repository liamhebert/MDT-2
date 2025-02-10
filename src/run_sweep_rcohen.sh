#!/bin/bash

nohup python train.py experiment=pretrain_siglip logger=wandb env=rcohen hparams_search=default &
