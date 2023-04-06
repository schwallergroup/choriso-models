export MKL_SERVICE_FORCE_INTEL=1
export PYTHONPATH=$PYTHONPATH:~/reaction_forward

cd OpenNMT_Transformer

onmt_build_vocab -config run_config.yaml \
    -src_seq_length 1000 -tgt_seq_length 1000 \
    -src_vocab_size 1000 -tgt_vocab_size 1000 \
    -n_sample -1
