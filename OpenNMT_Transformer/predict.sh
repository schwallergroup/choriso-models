export PYTHONPATH=$PYTHONPATH:/rwthfs/rz/cluster/home/iz782675/reaction_forward

cd OpenNMT_Transformer

onmt_translate -model runs/models/model_step_400000.pt -gpu 0 \
    --src ../data/cjhif/val-src.txt \
    --tgt ../data/cjhif/val-tgt.txt \
    --output runs/models/cjhif_model_step_400000_val_predictions.txt \
    --n_best 5 --beam_size 10 --max_length 300 --batch_size 64