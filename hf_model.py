import argparse
import os
import json
import numpy as np
import pandas as pd
import tempfile
from functools import reduce
from fnmatch import fnmatch
from sklearn.metrics import top_k_accuracy_score

import evaluate
from tokenizers import models as tokenizer_models
from tokenizers.trainers import WordLevelTrainer
from tokenizers import Regex, Tokenizer, pre_tokenizers, processors
from transformers.integrations import WandbCallback, CodeCarbonCallback
from transformers import AutoConfig, AutoTokenizer, EncoderDecoderConfig, PreTrainedTokenizerFast, BertConfig, \
    AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, \
    HfArgumentParser, BertGenerationConfig, PretrainedConfig
from transformers import Pipeline as HFPipeline


from benchmark_models import ReactionModel, BenchmarkPipeline
from model_args import ReactionModelArgs
from utils import prepare_data, ReactionForwardDataset, overwrite_config_with_tokenizer, top_k_accuracy, remove_spaces


class HuggingFaceArgs(ReactionModelArgs):

    def __init__(self):
        super().__init__()
        pass

    def preprocess_args(self):
        parser = argparse.ArgumentParser(description='preprocess')
        parser.add_argument("--data_dir", help="Dataset directory. Should contain the following files: train.tsv,"
                                               " val.tsv and test.tsv. Each file should have a column named canonic_rxn"
                                               " which is then used to split reactions into reactants and products",
                            type=str, default="")
        return parser

    def training_args(self):
        # TODO Think about how to make this more flexible. maybe use parent config class to distinguish if we have
        #  encoder-decoder or already built llm?
        bert_config_args = BertConfig()
        pretrained_config_args = PretrainedConfig()
        parser = HfArgumentParser([PretrainedConfig(), Seq2SeqTrainingArguments], prog="train")
        print(bert_config_args)
        print(pretrained_config_args)
        breakpoint()
        return parser

    def predict_args(self):
        parser = HfArgumentParser(prog="predict")
        return parser


class HuggingFaceTransformer(ReactionModel):

    def __init__(self, model_architecture: str, train_args: dict, model_args: dict = None, dataset: str = "cjhif"):
        name_suffix = model_architecture.split("/")[-1]
        self.name = f"HuggingFaceTransformer_{name_suffix}"
        super().__init__()

        self.args = HuggingFaceArgs()

        chem_tokenizer_path = os.path.join(self.model_dir, "chem_tokenizer")
        if not os.path.exists(chem_tokenizer_path):
            # TODO wip hardcoded MolecularTransformer tokenization
            pattern = Regex(
                r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])")
            # define simple Tokenizer that creates the vocabulary from the regex pattern
            chem_tokenizer = Tokenizer(tokenizer_models.WordLevel(unk_token="[UNK]"))

            chem_tokenizer.pre_tokenizer = pre_tokenizers.Split(pattern, "isolated")

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
                                                         model_max_length=2048,
                                                         padding_side="right")

                chem_tokenizer.post_processor = processors.BertProcessing(cls=("[CLS]", chem_tokenizer.cls_token_id),
                                                                          sep=("[SEP]", chem_tokenizer.sep_token_id))

                chem_tokenizer.save_pretrained(chem_tokenizer_path)

        else:
            # use pre-trained tokenizer
            chem_tokenizer = AutoTokenizer.from_pretrained(chem_tokenizer_path)

        chem_tokenizer.bos_token = chem_tokenizer.cls_token
        chem_tokenizer.eos_token = chem_tokenizer.sep_token

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

                enc_config = overwrite_config_with_tokenizer(enc_config, self.tokenizer)
                dec_config = overwrite_config_with_tokenizer(dec_config, self.tokenizer)

                dec_config.is_decoder = True
                dec_config.add_cross_attention = True

                model_kwargs.pop("encoder", None)
                model_kwargs.pop("decoder", None)

                config = EncoderDecoderConfig.from_encoder_decoder_configs(enc_config, dec_config, **model_kwargs)

                # overwrite arguments with tokenizer specifications
                config = overwrite_config_with_tokenizer(config, self.tokenizer)

                config.tie_word_embeddings = True

                print(config)

            else:
                # use pre-built architecture
                config = AutoConfig.from_pretrained(model_architecture, **model_kwargs)

            # save config for later runs
            config.save_pretrained(config_path, push_to_hub=False)

        # choose model based on config
        self.model = AutoModelForSeq2SeqLM.from_config(config)

        self.model.vocab_size = len(self.tokenizer)

        """# resize the token length to fit the tokenizer
        self.model.encoder.resize_token_embeddings(len(self.tokenizer))
        self.model.decoder.resize_token_embeddings(len(self.tokenizer))"""

        # route the given dir to the model dir
        train_args["output_dir"] = os.path.join(self.model_dir, train_args["output_dir"])
        train_args["logging_dir"] = os.path.join(self.model_dir, train_args["logging_dir"])
        self.train_args = Seq2SeqTrainingArguments(**train_args)

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
                # read and split data
                reactions = pd.read_csv(os.path.join(data_dir, f"{file_name}.tsv"), sep="\t")
                split_reactions = prepare_data(reactions, rsmiles_col="canonic_rxn")
                split_reactions = split_reactions.to_dict("list")

                # tokenize the dataset
                reaction_dataset = self.tokenizer(split_reactions["reactants"],
                                                  text_target=split_reactions["products"],
                                                  truncation=True, padding=True)  #, max_length=1000)

                # make it a dict
                reaction_dataset = dict(reaction_dataset)

                labels = np.array(reaction_dataset["labels"])
                labels = np.where(labels != self.tokenizer.pad_token_id, labels, -100)

                reaction_dataset["labels"] = labels.tolist()

                # save results
                with open(save_path, "w") as outfile:
                    json.dump(reaction_dataset, outfile)

            if file_name == "train":
                self.train_dataset = ReactionForwardDataset(reaction_dataset)
            else:
                self.val_dataset = ReactionForwardDataset(reaction_dataset)

    def compute_metrics(self, eval_preds):
        labels = eval_preds.label_ids
        preds = eval_preds.predictions

        labels[labels == -100] = self.tokenizer.pad_token_id

        def remove_special_tokens(line):
            mask = np.isin(line, self.tokenizer.all_special_ids)
            line = line[~mask]
            return line

        accuracy = evaluate.load("accuracy")
        sum = 0
        for p, l in zip(preds, labels):
            print(f"prediction: {self.tokenizer.decode(p, skip_special_tokens=True)}\n label: {self.tokenizer.decode(l, skip_special_tokens=True)}\n")
            l = remove_special_tokens(l)
            p = remove_special_tokens(p)

            if len(l) > len(p):
                # pad p
                p = np.pad(p, (0, int(len(l)-len(p))), "constant")
                size = len(l)
            elif len(l) < len(p):
                # pad l
                l = np.pad(l, (0, int(len(p)-len(l))), "constant")
                size = len(p)
            else:
                # same length, don't pad
                size = len(l)
            acc = accuracy.compute(references=l, predictions=p)["accuracy"]
            print(f"prediction: {p}\n label: {l}\n")
            print("accuracy: ", acc)
            sum += acc

        result = {"accuracy": (sum / preds.shape[0])}

        return result

    def train(self):
        """Train the reaction model. Should also contain validation and test steps"""
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)

        trainer = Seq2SeqTrainer(model=self.model,
                                 args=self.train_args,
                                 train_dataset=self.train_dataset,
                                 eval_dataset=self.val_dataset,
                                 data_collator=data_collator,
                                 # compute_metrics=self.compute_metrics,
                                 callbacks=[WandbCallback, CodeCarbonCallback])
        trainer.train()

    def predict(self, dataset: str = "cjhif", num_seq: int = 5):
        """Predict provided data with the reaction model"""
        # get test data path
        root_dir = os.path.dirname(self.model_dir)
        data_dir = os.path.join(root_dir, "data", dataset)
        test_data = os.path.join(data_dir, "test.tsv")

        # prepare data
        reactions = pd.read_csv(test_data, sep="\t")
        split_reactions = prepare_data(reactions, rsmiles_col="canonic_rxn").to_dict("list")

        inputs = split_reactions["reactants"]
        targets = split_reactions["products"]

        checkpoint_dir = os.path.join(self.model_dir, self.train_args.output_dir)

        preds = []
        # check if there are saved models in the dir
        for dir in os.listdir(checkpoint_dir):
            dir = os.path.join(checkpoint_dir, dir)
            if "checkpoint-5000" in dir and os.path.isdir(dir):
                model = AutoModelForSeq2SeqLM.from_pretrained(dir)
                input_ids = self.tokenizer(inputs[:10], padding=True, truncation=True, return_tensors="pt")["input_ids"]
                target_ids = self.tokenizer(targets[:10], padding=True, truncation=True, return_tensors="pt")["input_ids"]
                print("self.tokenizer.pad_token_id: ", self.tokenizer.pad_token_id)
                print("self.tokenizer.eos_token_id: ", self.tokenizer.eos_token_id)
                print("self.tokenizer.cls_token_id: ", self.tokenizer.cls_token_id)
                print("self.tokenizer.bos_token_id: ", self.tokenizer.bos_token_id)
                print("input_ids[0]: ", input_ids[0])
                print("target_ids[0]: ", target_ids[0])

                beam_outputs = model.generate(input_ids,
                                              max_length=96,
                                              num_beams=10,
                                              pad_token_id=self.tokenizer.pad_token_id,
                                              eos_token_id=self.tokenizer.eos_token_id,
                                              decoder_start_token_id=self.tokenizer.cls_token_id,
                                              bos_token_id=self.tokenizer.bos_token_id,
                                              num_return_sequences=num_seq,
                                              early_stopping=True)

                pred_smiles = self.tokenizer.batch_decode(beam_outputs.tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=True)
                pred_smiles = np.array(pred_smiles).reshape(-1, num_seq)
                pred_smiles = np.vectorize(remove_spaces)(pred_smiles)
                print("pred_smiles.shape: ", pred_smiles.shape)
                for target_smiles, temp_smiles in zip(targets[:10], pred_smiles.tolist()):
                    print("Target: ", target_smiles)
                    print("\n")

                    print("Prediction: ", temp_smiles)
                    print("\n")

                top_k_acc = top_k_accuracy(pred_smiles.tolist(), targets[:10], k=5)
                print("top_k_acc: ", top_k_acc)
                preds.append(pred_smiles)

        # print("preds: ", preds)

        # top_k_accs = top_k_accuracy(preds, targets, k=5)
        # print("top_k_accs: ", top_k_accs)


if __name__ == "__main__":
    train_args = {
        "output_dir": 'results',  # output directory

        # training setup
        "max_steps": 150000,  # total number of training steps
        "evaluation_strategy": "steps",
        "eval_steps": 5000,
        "save_strategy": "steps",
        "save_steps": 5000,

        # model and optimizer params
        "learning_rate": 5.5e-4,
        "save_total_limit": 3,
        "per_device_train_batch_size": 72,  # batch size per device during training
        "per_device_eval_batch_size": 96,  # batch size for evaluation
        "warmup_steps": 5000,  # number of warmup steps for learning rate scheduler
        "weight_decay": 0.01,  # strength of weight decay
        "logging_dir": 'logs',  # directory for storing logs
        "logging_steps": 10,
        "adam_beta1": 0.9,
        "adam_beta2": 0.998,
        "max_grad_norm": 0,
        "label_smoothing_factor": 0,
        "gradient_accumulation_steps": 4,
        "eval_accumulation_steps": 10,
        "predict_with_generate": True,
        "dataloader_num_workers": 8,
    }
    """config = BertGenerationConfig(hidden_size=384,
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
                                  position_embedding_type='absolute',
                                  use_cache=True,
                                  classifier_dropout=None,
                                  num_beams=10,
                                  vocab_size=1024,
                                  max_length=96,
                                  min_length=1)

    model_args = {
        "encoder": config,
        "decoder": config,
        "length_penalty": 0,
        "max_length": 96,
        "min_length": 3
    }
    reaction_model = HuggingFaceTransformer(model_architecture="bert-base-uncased",
                                            train_args=train_args, model_args=model_args)"""
    model_args = {"d_model": 384,
                  "encoder_layers": 4,
                  "decoder_layers": 4,
                  "encoder_attention_heads": 8,
                  "decoder_attention_heads": 8,
                  "encoder_ffn_dim": 2048,
                  "decoder_ffn_dim": 2048,
                  "activation_function ": 'gelu',
                  "dropout": 0.1,
                  "attention_dropout ": 0.1,
                  "max_position_embeddings": 512,
                  "layer_norm_eps": 1e-12,
                  "position_embedding_type": 'absolute',
                  "use_cache": True,
                  "num_beams": 10,
                  "vocab_size": 1024,
                  "max_length": 96,
                  "min_length": 1}

    reaction_model = HuggingFaceTransformer(model_architecture="facebook/bart-large",
                                            train_args=train_args, model_args=model_args)

    pipeline = BenchmarkPipeline(model=reaction_model)
    pipeline.run_train_pipeline()
    # pipeline.predict(dataset="cjhif")
