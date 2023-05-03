#!/bin/bash

#PBS -N myjob
#PBS -l nodes=1:ppn=4
#PBS -l walltime=48:00:00

module load cuda
module load intel

export PATH=~/anaconda3/envs/reaction_prediction/bin:$PATH
# conda env
source activate reaction_prediction

models=(Graph2SMILES OpenNMT_Transformer)
datasets=(cjhif choriso_low_mw choriso_high_mw)

for model in "${models[@]}"
do
  for dataset in "${datasets[@]}"
  do
    qsub -v model=$model,dataset=$dataset myjob.sh
  done
done