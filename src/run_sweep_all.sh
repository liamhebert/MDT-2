#!/bin/bash

nohup python train.py experiment=pretrain_siglip logger=wandb env=all hparams_search=default &
