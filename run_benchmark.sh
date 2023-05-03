models=(G2S ONMT)
datasets=(cjhif choriso_low_mw choriso_high_mw)

for model in "${models[@]}"
do
  for dataset in "${datasets[@]}"
  do
    sbatch --export=model=$model,dataset=$dataset run_main.sh
  done
done
