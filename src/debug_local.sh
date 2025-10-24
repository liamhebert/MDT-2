#!/bin/sh
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --partition=ALL
#SBATCH --mail-user=l2hebert@uwaterloo.ca
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --signal=SIGUSR1@90
#SBATCH --nodelist=watgpu508

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=0

# srun python train.py experiment=giga_pretrain_roberta_sup_con logger=wandb env=all hparams_search=pretrain hydra/launcher=basic hydra.sweeper.n_trials=1 trainer=gpu trainer.max_epochs=3 trainer.min_epochs=3 dataset.dataset.debug=40
srun --ntasks-per-node=1 uv run train.py \
    experiment=pretrain_v3_gemma_contrastive \
    logger=wandb \
    env=all \
    hparams_search=pretrain_v3 \
    hydra/launcher=basic \
    trainer=gpu \
    trainer.max_epochs=10 \
    trainer.min_epochs=1 \
    trainer.devices=1 \
    dataset.group_size=2 \
    dataset.train_batch_size=10 \
    dataset.test_batch_size=10 \
    model.encoder.block_size=1 \
    callbacks=model_summary


# dataset.dataset.debug=2
