#!/bin/bash

#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=50G

#request one volta gpus (CLAIX18)
#SBATCH --gres=gpu:volta:1

#SBATCH --time=0-72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=malte.franke@rwth-aachen.de

### begin of executable commands
module load cuda
module load intel

export PATH=~/anaconda3/envs/reaction_prediction/bin:$PATH
# conda env
source activate reaction_prediction

export PYTHONPATH=$PYTHONPATH:~/reaction_forward


python main.py -model $model -m tp -d $dataset
