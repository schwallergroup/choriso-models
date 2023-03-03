export PYTHONPATH=$PYTHONPATH:/rwthfs/rz/cluster/home/iz782675/reaction_forward
export PROJECT_ROOT=../megan
export DATA_DIR=../data

cd megan

export CONFIGS_DIR=./configs
export LOGS_DIR=./logs
export MODELS_DIR=./models

python bin/train.py cjhif ../megan/models/cjhif

                                
