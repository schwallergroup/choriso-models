export PYTHONPATH=$PYTHONPATH:/rwthfs/rz/cluster/home/iz782675/reaction_forward

cd OpenNMT_Transformer

onmt_translate -model runs/models/cjhif_step_500.pt -gpu 0 \
    --src ../data/cjhif/src-test.txt \
    --tgt ../data/cjhif/tgt-test.txt \
    --output runs/models/cjhif_model_step_500_test_predictions.txt \
    --n_best 5 --beam_size 10 --max_length 300 --batch_size 64