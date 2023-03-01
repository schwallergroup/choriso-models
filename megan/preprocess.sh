export PYTHONPATH=$PYTHONPATH:/rwthfs/rz/cluster/home/iz782675/reaction_forward
export PROJECT_ROOT=reaction_forward
export DATA_DIR=$PROJECT_ROOT/data

export CONFIGS_DIR=./configs
export LOGS_DIR=./logs
export MODELS_DIR=./models

cd megan

python bin/featurize.py cjhik megan_16_bfs_randat
