#!/bin/bash

#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=50G

#request one gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --qos=gpu

#SBATCH --time=0-96:00:00
#SBATCH --mail-type=ALL

### begin of executable commands
module load cuda
module load intel

export PATH=~/anaconda3/envs/reaction_prediction/bin:$PATH
# conda env
source activate reaction_prediction

export PYTHONPATH=$PYTHONPATH:~/rxn_predict


python main.py -model $model -p tp -ds $dataset
