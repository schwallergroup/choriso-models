export PYTHONPATH=$PYTHONPATH:~/reaction_forward

export PATH=~/anaconda3/envs/reaction_prediction/bin:$PATH
# conda env
source activate reaction_prediction

cd DiffuSeq
python -m torch.distributed.launch --nproc_per_node=4 --master_port=12233 --use_env run_train.py \
--diff_steps 2000 \
--lr 0.0001 \
--learning_steps 80000 \
--save_interval 10000 \
--seed 102 \
--noise_schedule sqrt \
--hidden_dim 128 \
--bsz 2048 \
--dataset qqp \
--data_dir ./processed/cjhif \
--vocab bert \
--seq_len 128 \
--schedule_sampler lossaware \
--notes cjhif
