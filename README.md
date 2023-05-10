# Reaction forward benchmark models
This repository is released in addition to the [choriso dataset](https://github.com/schwallergroup/choriso) and provides functionality to benchmark models.



## Setup
To install and activate the environment, run the following commands:
```
conda env create -f env.yml
conda activate reaction_prediction
```

Before you can train a model, you have to transfer the data into this repo. 
Please follow the steps below to make sure your data is processed correctly:
1. Follow the steps in [choriso](https://github.com/schwallergroup/choriso) to create data splits **train.tsv**, **val.tsv** and **test.tsv**. 
2. Alternatively, create data splits yourself with the file names above. Please name the reaction SMILES column "**canonic_rxn**".
3. Transfer the data to **data/<dataset_name>** 

## Running the benchmark
To run single models, you can run the following commands:
```
python main.py -m [model] -p [phase] -ds [dataset]
```
model: one of ('Graph2SMILES', 'G2S', 'g2s', 'graph2smiles') or ('OpenNMT', 'ONMT', 'onmt', 'opennmt') <br />
phase: either t (preprocessing + training), p (predicting) or tp (preprocessing + training + predicting) <br />
dataset: name of your dataset. data has to be put into data/<dataset_name> as described above <br />
  

Assuming you have the choriso splits in data/ and have a slurm system available, the whole benchmarking study can be submitted using 
```
./run_benchmark.sh
```
You can also modify this file to take in your datasets for automatic benchmarking.  
