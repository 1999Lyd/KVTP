#!/usr/bin/bash
#SBATCH --job-name=train_siglip
#SBATCH --cpus-per-task 32
#SBATCH --gpus 1
#SBATCH --mem 128g
#SBATCH -e train_siglip_contra.err
#SBATCH -o train_siglip_contra.out
#SBATCH --partition athena-genai
#SBATCH --account yl817

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export WORLD_SIZE=1

torchrun --nproc_per_node=1 train_siglip.py #> train_klloss_0.9_softmax_pool.log 2>&1 &