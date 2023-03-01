export PYTHONPATH=$PYTHONPATH:/rwthfs/rz/cluster/home/iz782675/reaction_forward

cd MolecularTransformer

dataset=cjhif

## build the directory if it doesn't yet exist 
mkdir -p processed/${dataset}/

python preprocess.py -train_src ../data/${dataset}/src-train.txt \
                     -train_tgt ../data/${dataset}/tgt-train.txt \
                     -valid_src ../data/${dataset}/src-val.txt \
                     -valid_tgt ../data/${dataset}/tgt-val.txt \
                     -save_data processed/${dataset}/ \
                     -src_seq_length 1000 -tgt_seq_length 1000 \
                     -src_vocab_size 1000 -tgt_vocab_size 1000 -share_vocab
