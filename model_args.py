import abc
import argparse

from transformers import Seq2SeqTrainingArguments, HfArgumentParser, BertConfig, PretrainedConfig

from Graph2SMILES.preprocess import get_preprocess_parser
from Graph2SMILES.train import get_train_parser
from Graph2SMILES.predict import get_predict_parser

from onmt.opts import train_opts, translate_opts, dynamic_prepare_opts, config_opts


class ReactionModelArgs(abc.ABC):
    """Abstract class for wrapping arguments for all stages of model training. Inspired by the Graph2SMILES args code"""
    def __init__(self):
        pass

    def preprocess_args(self):
        pass

    def training_args(self):
        pass

    def predict_args(self):
        pass


class G2SArgs(ReactionModelArgs):

    def __init__(self):
        super().__init__()
        pass

    def preprocess_args(self):
        return get_preprocess_parser()

    def training_args(self):
        return get_train_parser()

    def predict_args(self):
        return get_predict_parser()


class OpenNMTArgs(ReactionModelArgs):

    def __init__(self):
        super().__init__()
        pass

    def preprocess_args(self):
        parser = argparse.ArgumentParser(description='preprocess')
        dynamic_prepare_opts(parser, build_vocab_only=True)
        return parser

    def training_args(self):
        parser = argparse.ArgumentParser(description='train')
        train_opts(parser)
        return parser

    def predict_args(self):
        parser = argparse.ArgumentParser(description='predict')
        config_opts(parser)
        translate_opts(parser, dynamic=True)
        return parser


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


if __name__ == "__main__":

    test_args = HuggingFaceArgs()
    parser = test_args.training_args()
    args = parser.parse_args()

    print(args)
