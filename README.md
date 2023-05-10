# Reaction forward benchmark models
This repository is released in addition to the [choriso dataset](https://github.com/schwallergroup/choriso) and provides functionality to benchmark models.
We focus on the MolecularTransformer([Schwaller et al. 2019](https://pubs.acs.org/doi/full/10.1021/acscentsci.9b00576)) and Graph2SMILES([Tu et al. 2022](https://pubs.acs.org/doi/full/10.1021/acs.jcim.2c00321)) and study their performance, also regarding sustainability aspects.


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
python main.py -m [model] -p [phase] -ds [dataset_name]
```
model: one of ('Graph2SMILES', 'G2S', 'g2s', 'graph2smiles') or ('OpenNMT', 'ONMT', 'onmt', 'opennmt') <br />
phase: either t (preprocessing + training), p (predicting) or tp (preprocessing + training + predicting) <br />
dataset_name: name of your dataset. data has to be put into data/<dataset_name> as described above <br />
  

Assuming you have the choriso splits in data/ and have a slurm system available, the whole benchmarking study can be submitted using 
```
./run_benchmark.sh
```
This loops over specified models and dataset to submit jobs to your gpu cluster. 
You can modify this file to take in your datasets for automatic benchmarking.  
Changes to the submission settings can be made in ```run_main.sh```.
