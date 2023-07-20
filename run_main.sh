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
module load cuda
module load intel

export PATH=~/anaconda3/envs/choriso-models/bin:$PATH
# conda env
source activate choriso-models

export PYTHONPATH=$PYTHONPATH:~/choriso-models


python main.py -model $model -p tp -ds $dataset
