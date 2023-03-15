import abc
import argparse

from transformers import Seq2SeqTrainingArguments, HfArgumentParser

from Graph2SMILES.preprocess import get_preprocess_parser
from Graph2SMILES.train import get_train_parser
from Graph2SMILES.predict import get_predict_parser


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
        pass

    def training_args(self):
        pass

    def predict_args(self):
        pass


class HuggingfaceArgs(ReactionModelArgs):

    def __init__(self):
        super().__init__()
        pass

    def preprocess_args(self):
        parser = HfArgumentParser(prog="preprocess")
        return parser

    def training_args(self):
        parser = HfArgumentParser([Seq2SeqTrainingArguments], prog="train")
        return parser

    def predict_args(self):
        parser = HfArgumentParser(prog="predict")
        return parser


if __name__=="__main__":

    test_args = HuggingfaceArgs()
    parser = test_args.training_args()
    args = parser.parse_args()

    print(args)
