#!/bin/bash

nohup python train.py experiment=pretrain_siglip_single logger=wandb env=all_single hparams_search=pretrain &
