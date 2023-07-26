#!/bin/bash

#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=80G

#request one gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --qos=gpu

#SBATCH --time=0-72:00:00
#SBATCH --mail-type=ALL

### begin of executable commands
module load intel/19.0.5
module load cuda/11.6.2

export PATH=~/anaconda3/envs/choriso-models/bin:$PATH
# conda env
source activate choriso-models

export PYTHONPATH=$PYTHONPATH:$SLURM_SUBMIT_DIR


WANDB_MODE=disabled python main.py -model $model -p tp -ds $dataset
