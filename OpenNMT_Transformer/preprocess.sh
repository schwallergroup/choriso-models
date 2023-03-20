export PYTHONPATH=$PYTHONPATH:/rwthfs/rz/cluster/home/iz782675/reaction_forward

###cd OpenNMT_Transformer

onmt_build_vocab -config OpenNMT_Transformer/run_config.yaml \
    -src_seq_length 1000 -tgt_seq_length 1000 \
    -src_vocab_size 1000 -tgt_vocab_size 1000 \
    -n_sample -1