export PYTHONPATH=$PYTHONPATH:/rwthfs/rz/cluster/home/iz782675/reaction_forward

cd OpenNMT_Transformer

onmt_train -config run_config.yaml \
        -seed 42 -gpu_ranks 0  \
        -param_init 0 \
        -param_init_glorot -max_generator_batches 32 \
        -batch_type tokens -batch_size 6144\
         -normalization tokens -max_grad_norm 0  -accum_count 4 \
        -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam  \
        -warmup_steps 8000 -learning_rate 2 -label_smoothing 0.0 \
        -layers 4 -hidden_size  384 -word_vec_size 384 \
        -encoder_type transformer -decoder_type transformer \
        -dropout 0.1 -position_encoding -share_embeddings  \
        -global_attention general -global_attention_function softmax \
        -self_attn_type scaled-dot -heads 8 -transformer_ff 2048 \
        -tensorboard True -tensorboard_log_dir log_dir