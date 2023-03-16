import os
import sys
import torch
import wandb
import pandas as pd
import json
import numpy as np
import tempfile
from tokenizers import Regex, Tokenizer, pre_tokenizers, processors
from tokenizers import models as tokenizer_models
from tokenizers.trainers import WordLevelTrainer
from transformers import AutoConfig, AutoTokenizer, EncoderDecoderConfig, PreTrainedTokenizerFast, BertConfig
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import evaluate


from transformers.integrations import WandbCallback, CodeCarbonCallback

from reaction_model import *
from utils import prepare_data, ReactionForwardDataset


class MolecularTransformer(ReactionModel):

    def __init__(self):
        self.name = "MolecularTransformer"
        super().__init__()

    def embed(self):
        """Get the embedding of the reaction model"""
        pass

    def preprocess(self):
        """Do data preprocessing. Skip if preprocessed data already exists"""
        os.system("sh MolecularTransformer/preprocess.sh")

    def train(self):
        """Train the reaction model. Should also contain validation and test steps"""
        os.system("sh MolecularTransformer/train.sh")

    def predict(self, data):
        """Predict provided data with the reaction model"""
        pass


class OpenNMT(ReactionModel):

    def __init__(self):
        self.name = "OpenNMT_Transformer"
        super().__init__()

    def embed(self):
        """Get the embedding of the reaction model"""
        pass

    def preprocess(self):
        """Do data preprocessing. Skip if preprocessed data already exists"""
        os.system("sh OpenNMT_Transformer/preprocess.sh")

    def train(self):
        """Train the reaction model. Should also contain validation and test steps"""
        # wandb.init(project="OpenNMT_Transformer")
        # wandb.tensorboard.patch(root_logdir=os.path.join(self.model_dir, "log_dir"))
        os.system("sh OpenNMT_Transformer/train.sh")
        # wandb.finish()

    def predict(self, data):
        """Predict provided data with the reaction model"""
        os.system("sh OpenNMT_Transformer/predict.sh")


class HuggingFaceTransformer(ReactionModel):

    def __init__(self, model_architecture: str, train_args: dict, model_args: dict = None, dataset: str = "cjhif"):
        name_suffix = model_architecture.split("/")[-1]
        self.name = f"HuggingFaceTransformer_{name_suffix}"
        super().__init__()
        chem_tokenizer_path = os.path.join(self.model_dir, "chem_tokenizer")
        if not os.path.exists(chem_tokenizer_path):
            # TODO wip hardcoded MolecularTransformer tokenization
            pattern = Regex(
                r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])")
            # define simple Tokenizer that creates the vocabulary from the regex pattern
            chem_tokenizer = Tokenizer(tokenizer_models.WordLevel(unk_token="[UNK]"))

            chem_tokenizer.pre_tokenizer = pre_tokenizers.Split(pattern, "isolated")
            chem_tokenizer.post_processor = processors.BertProcessing(cls=("[CLS]", 1), sep=("[SEP]", 2))
            trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"])

            # get data for tokenizer training --> for building the vocab!
            root_dir = os.path.dirname(self.model_dir)
            data_dir = os.path.join(root_dir, "data", dataset)
            paths = [os.path.join(data_dir, f"{file_name}.tsv") for file_name in ["train", "val"]]

            # obtain reaction smiles and merge them into one dataframe
            reactions = [pd.read_csv(path, sep="\t")["canonic_rxn"] for path in paths]
            reactions = pd.concat(reactions)

            # open a temporary file for the train+val smiles so it doesn't need to be stored
            with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
                # save to temp file
                reactions.to_csv(tmp, sep="\t", index=False, header=False)

                # do training and store the final tokenizer
                chem_tokenizer.train([tmp.name], trainer)
                chem_tokenizer = PreTrainedTokenizerFast(tokenizer_object=chem_tokenizer,
                                                         unk_token="[UNK]",
                                                         pad_token="[PAD]",
                                                         cls_token="[CLS]",
                                                         sep_token="[SEP]",
                                                         mask_token="[MASK]",
                                                         model_max_length=1024)

                chem_tokenizer.save_pretrained(chem_tokenizer_path)

        else:
            # use pre-trained tokenizer
            chem_tokenizer = AutoTokenizer.from_pretrained(chem_tokenizer_path)

        self.tokenizer = chem_tokenizer

        # if we already have saved a model config, load it
        config_path = os.path.join(self.model_dir, "config")
        if os.path.exists(config_path):
            print("WARNING: Saved parameters are used! If you want to change the architecture, please delete the "
                  "config file.")
            config = AutoConfig.from_pretrained(config_path)

        else:
            model_kwargs = {} if model_args is None else model_args
            # we need to decide which architecture to use. We could either use pre-defined ones like T5 or specify a
            # encoder-decoder model ourselves. In this case we need to specify encoder and decoder configs

            if "encoder" in model_kwargs.keys() and "decoder" in model_kwargs.keys():
                # use encoder-decoder architecture with given parameters
                enc_config = model_kwargs["encoder"]
                dec_config = model_kwargs["decoder"]

                config = EncoderDecoderConfig.from_encoder_decoder_configs(enc_config, dec_config)

                # overwrite arguments with tokenizer specifications
                config.decoder_start_token_id = self.tokenizer.cls_token_id
                config.pad_token_id = self.tokenizer.pad_token_id
                config.vocab_size = self.tokenizer.vocab_size

                config.tie_word_embeddings = True

            else:
                # use pre-built architecture
                config = AutoConfig.from_pretrained(model_architecture, **model_kwargs)

            # save config for later runsl
            config.save_pretrained(config_path, push_to_hub=False)

        # choose model based on config
        self.model = AutoModelForSeq2SeqLM.from_config(config)

        # resize the token length to fit the tokenizer
        self.model.encoder.resize_token_embeddings(len(self.tokenizer))
        self.model.decoder.resize_token_embeddings(len(self.tokenizer))

        # route the given dir to the model dir
        train_args["output_dir"] = os.path.join(self.model_dir, train_args["output_dir"])
        train_args["logging_dir"] = os.path.join(self.model_dir, train_args["logging_dir"])
        self.train_args = Seq2SeqTrainingArguments(**train_args)

    def embed(self):
        """Get the embedding of the reaction model"""
        pass

    def compute_metrics(self, eval_pred):
        metric = evaluate.load("accuracy")
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    def preprocess(self, dataset: str = "cjhif"):
        """Do data preprocessing. Skip if preprocessed data already exists"""
        root_dir = os.path.dirname(self.model_dir)

        preprocessed_dir = os.path.join(self.model_dir, "preprocessed")
        if not os.path.exists(preprocessed_dir):
            os.mkdir(preprocessed_dir)

        data_dir = os.path.join(root_dir, "data", dataset)

        for file_name in ["val", "train"]:
            save_path = os.path.join(preprocessed_dir, f"{file_name}_dataset.json")

            # if there is saved data, load it
            if os.path.exists(save_path):
                with open(save_path, "r") as f:
                    reaction_dataset = json.load(f)

            # else do preprocessing and save it in a json file so it can be retrieved more easily
            else:
                reactions = pd.read_csv(os.path.join(data_dir, f"{file_name}.tsv"), sep="\t")

                split_reactions = prepare_data(reactions, rsmiles_col="canonic_rxn")

                split_reactions = split_reactions.to_dict("list")

                # tokenize the dataset
                reaction_dataset = self.tokenizer(split_reactions["reactants"], text_target=split_reactions["products"],
                                               truncation=True, padding=True, max_length=1000)

                # make it a dict
                reaction_dataset = dict(reaction_dataset)

                # save results
                with open(save_path, "w") as outfile:
                    json.dump(reaction_dataset, outfile)

            if file_name == "train":
                self.train_dataset = ReactionForwardDataset(reaction_dataset)
            else:
                self.val_dataset = ReactionForwardDataset(reaction_dataset)

    def train(self):
        """Train the reaction model. Should also contain validation and test steps"""
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)
        trainer = Seq2SeqTrainer(model=self.model,
                                 args=self.train_args,
                                 train_dataset=self.train_dataset,
                                 eval_dataset=self.val_dataset,
                                 data_collator=data_collator,
                                 tokenizer=self.tokenizer,
                                 # compute_metrics=self.compute_metrics,
                                 callbacks=[WandbCallback, CodeCarbonCallback])
        trainer.train()

    def predict(self, data):
        """Predict provided data with the reaction model"""
        pass


class G2S(ReactionModel):

    def __init__(self):
        self.name = "Graph2SMILES"
        super().__init__()

    def embed(self):
        """Get the embedding of the reaction model"""
        pass

    def preprocess(self):
        """Do data preprocessing. Skip if preprocessed data already exists"""
        os.system("sh Graph2SMILES/scripts/preprocess.sh")

    def train(self):
        """Train the reaction model. Should also contain validation and test steps"""
        os.system("sh Graph2SMILES/scripts/train_g2s.sh")

    def predict(self, data):
        """Predict provided data with the reaction model"""
        os.system("sh Graph2SMILES/scripts/predict.sh")


class MEGAN(ReactionModel):

    def __init__(self):
        self.name = "MEGAN"
        super().__init__()

    def embed(self):
        """Get the embedding of the reaction model"""
        pass

    def preprocess(self):
        """Do data preprocessing. Skip if preprocessed data already exists"""
        os.system("sh megan/preprocess.sh")

    def train(self):
        """Train the reaction model. Should also contain validation and test steps"""
        os.system("sh megan/train.sh")

    def predict(self, data):
        """Predict provided data with the reaction model"""
        pass


class BenchmarkPipeline:

    def __init__(self, model: ReactionModel):
        self.model = model

    def run_train_pipeline(self):
        self.model.preprocess()
        self.model.train()

    def predict(self):
        self.model.predict(data=None)


if __name__ == "__main__":
    train_args = {
        "output_dir": 'results',  # output directory

        # training setup
        "max_steps": 400000,  # total number of training steps
        "evaluation_strategy": "steps",
        "eval_steps": 10000,
        "save_strategy": "steps",
        "save_steps": 5000,

        # model and optimizer params
        "learning_rate": 2,
        "save_total_limit": 4,
        "per_device_train_batch_size": 8,  # batch size per device during training
        "per_device_eval_batch_size": 8,  # batch size for evaluation
        "warmup_steps": 8000,  # number of warmup steps for learning rate scheduler
        "weight_decay": 0.01,  # strength of weight decay
        "logging_dir": 'logs',  # directory for storing logs
        "logging_steps": 10,
        "adam_beta1": 0.9,
        "adam_beta2": 0.998,
        "max_grad_norm": 0,
        "label_smoothing_factor": 0,
        # "load_best_model_at_end": False,
    }
    enc_config = BertConfig(hidden_size = 384,
                              num_hidden_layers = 4,
                              num_attention_heads = 8,
                              intermediate_size = 2048,
                              hidden_act = 'gelu',
                              hidden_dropout_prob = 0.1,
                              attention_probs_dropout_prob = 0.1,
                              max_position_embeddings = 512,
                              type_vocab_size = 2,
                              initializer_range = 0.02,
                              layer_norm_eps = 1e-12,
                              pad_token_id = 0,
                              position_embedding_type = 'absolute',
                              use_cache = True,
                              classifier_dropout = None,
                            num_beams=10, vocab_size=1024)

    dec_config = BertConfig(hidden_size=384,
                            num_hidden_layers=4,
                            num_attention_heads=8,
                            intermediate_size=2048,
                            hidden_act='gelu',
                            hidden_dropout_prob=0.1,
                            attention_probs_dropout_prob=0.1,
                            max_position_embeddings=512,
                            type_vocab_size=2,
                            initializer_range=0.02,
                            layer_norm_eps=1e-12,
                            pad_token_id=0,
                            position_embedding_type='absolute',
                            use_cache=True,
                            classifier_dropout=None, num_beams=10, vocab_size=1024)

    model_args = {
        "encoder": enc_config,
        "decoder": dec_config
    }
    reaction_model = HuggingFaceTransformer(model_architecture="bert-base-uncased",
                                                  train_args=train_args, model_args=model_args)

    onmt_model = OpenNMT()
    g2s_model = G2S()

    pipeline = BenchmarkPipeline(model=reaction_model)
    pipeline.run_train_pipeline()
    # pipeline.predict()
