###!/usr/local_rwth/bin/zsh

#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=90G

#request one volta gpus (CLAIX18)
#SBATCH --gres=gpu:volta:1

#SBATCH --time=0-72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=malte.franke@rwth-aachen.de

### begin of executable commands
module load cuda
module load intel

export PATH=~/anaconda3/envs/reaction_pred/bin:$PATH
# conda env
source activate reaction_pred

export PYTHONPATH=$PYTHONPATH:~/reaction_forward


python benchmark_models.py
