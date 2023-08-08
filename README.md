# choriso-models: reaction prediction benchmark 
This repository is released in addition to the [choriso dataset](https://github.com/schwallergroup/choriso) and provides functionality to benchmark models.
We focus on the MolecularTransformer ([Schwaller et al. 2019](https://pubs.acs.org/doi/full/10.1021/acscentsci.9b00576)) and Graph2SMILES ([Tu et al. 2022](https://pubs.acs.org/doi/full/10.1021/acs.jcim.2c00321)) and study their performance, also regarding sustainability aspects.


## 🚀 Installation
To install and activate the environment, run the following commands:
```
conda env create -f env.yml
conda activate choriso-models
```

Before you can train a model, you have to transfer the data into this repo. 
Please follow the steps below to make sure your data is processed correctly:
1. Follow the steps in [choriso](https://github.com/schwallergroup/choriso) to create data splits.  
2. Alternatively, create data splits yourself with the file names above. Please name the reaction SMILES column "**canonic_rxn**".
3. Move the data split to ``data/<dataset_name>/`` 
4. Rename your files to **train.tsv**, **val.tsv** and **test.tsv** respectively.

## 🔥 Quick start
If you want to train the Molecular Transformer on the choriso dataset and predict on the test set, run:
```
python main.py -m onmt -p tp -ds choriso
```

##  :brain: Advanced usage
To run single models, you can run the following commands:
```
python main.py -m [model] -p [phase] -ds [dataset_name]
```
model:
* ('Graph2SMILES', 'G2S', 'g2s', 'graph2smiles') for [Graph2SMILES](https://github.com/coleygroup/Graph2SMILES) <br />
* ('OpenNMT', 'ONMT', 'onmt', 'opennmt') for [Molecular Transformer](https://github.com/pschwllr/MolecularTransformer) <br />

phase:
* t (preprocessing + training) <br />
* p (predicting) <br />
* tp (preprocessing + training + predicting) <br />

dataset_name: 
* name of your dataset. Data has to be put into ``data/<dataset_name>`` as described above <br />
  
For example, to train Graph2SMILES on choriso_low_mw and predict on the test set, run 
```
python main.py -m g2s -p tp -ds choriso_low_mw
```
In ``<model>/<dataset_name>/results/`` you can find the predictions (**all_results.csv**) as well as the sustainability evaluation.

## 🤖 Batch submission
Assuming you have the choriso splits in data/ and have a slurm system available, the whole benchmarking study can be submitted using 
```
./run_benchmark.sh
```
This loops over specified models (default: G2S, ONMT) and datasets (default: choriso, choriso_low_mw, choriso_high_mw) to submit jobs to your GPU cluster. 
You can modify this file to take in your datasets for automatic benchmarking.  
Submission settings can be changed in ```run_main.sh```.

## 🌳 Sustainability assessment
We use [eco2AI](https://github.com/sb-ai-lab/Eco2AI) to track carbon emissions and compute power usage. 
For correct tracking on, please specify your region in ```benchmark_models.py``` <br> 
⚠️ Be aware that eco2AI tracks **all** GPU emissions from your system. 
If your system has more than 1 GPUs and you run your code on only one of them, the reported GPU emissions for your run might be faulty (see this [issue](https://github.com/sb-ai-lab/Eco2AI/issues/11)). 
For this reason, we used a remote slurm system to request single GPUs that are then tracked correctly.


