import argparse
import os
import torch
from utils import set_pythonpath, prepare_parser
from g2s_model import G2SArgs, G2S
from onmt_model import OpenNMTArgs, OpenNMT
from hf_model import HuggingFaceArgs, HuggingFaceTransformer
from neuralsym_model import NeuralsymArgs, Neuralsym
from diffuseq_model import DiffuSeq
from benchmark_models import BenchmarkPipeline


def get_base_parsers():
    # Define the main parser
    parser = argparse.ArgumentParser(description='Model benchmarking')

    parser.add_argument('--dataset', aliases=['--ds'], type=str, default='cjhif',
                        help='Dataset to use for training and evaluation')

    # Define the subparsers for each model
    subparsers = parser.add_subparsers(title='Models', dest='model', required=True)

    parser_g2s = subparsers.add_parser('Graph2SMILES', aliases=['G2S', 'g2s', 'graph2smiles'],
                                       help='Graph2SMILES model')
    parser_onmt = subparsers.add_parser('OpenNMT', aliases=['ONMT', 'onmt', 'opennmt'], help='OpenNMT model')
    parser_hf = subparsers.add_parser('HuggingFace', aliases=['HF', 'hf', 'huggingface', 'Huggingface'],
                                      help='Huggingface model')
    parser_ns = subparsers.add_parser('Neuralsym', aliases=['NS', 'ns', 'neuralsym', 'NeuralSym'],
                                      help='Neuralsym model')

    return parser, {"G2S": {"base_parser": parser_g2s,
                            "args_class": G2SArgs()},
                    "ONMT": {"base_parser": parser_onmt,
                             "args_class": OpenNMTArgs()},
                    "NeuralSym": {"base_parser": parser_ns,
                                  "args_class": NeuralsymArgs()},}
                    # "HF": {"base_parser": parser_hf,
                    #       "args_class": HuggingFaceArgs()}}


def add_mode_subparser(model_parser):
    mode_subparser = model_parser.add_subparsers(title="Run mode", dest="mode", required=True)

    train_mode_parser = mode_subparser.add_parser('train', aliases=['t'], help='Training mode')
    predict_mode_parser = mode_subparser.add_parser('predict', aliases=['p', 'pred'], help='Prediction mode')

    return train_mode_parser, predict_mode_parser


def build_parser():

    parser, parser_dict = get_base_parsers()

    for model in parser_dict:
        model_base_parser = parser_dict[model]["base_parser"]
        model_args = parser_dict[model]["args_class"]

        train_parser, predict_parser = add_mode_subparser(model_base_parser)

        for train_arg in model_args.training_args()._actions:
            try:
                train_parser._add_action(train_arg)
            except:
                continue

        for pred_arg in model_args.predict_args()._actions:
            try:
                predict_parser._add_action(pred_arg)
            except:
                continue

        parser_dict[model]["train_parser"] = train_parser
        parser_dict[model]["predict_parser"] = predict_parser

    return parser


def main(args):

    # TODO this is quite static, make more dynamic
    # instantiate model depending on args
    if args.model in ["Graph2SMILES", 'G2S', 'g2s', 'graph2smiles']:
        reaction_model = G2S()
    elif args.model in ["OpenNMT", 'ONMT', 'onmt', 'opennmt']:
        reaction_model = OpenNMT()
    elif args.model in ["HuggingFace", 'HF', 'hf', 'huggingface', 'Huggingface']:
        reaction_model = HuggingFaceTransformer()
    elif args.model in ["Neuralsym", 'NS', 'ns', 'neuralsym', 'NeuralSym']:
        reaction_model = Neuralsym()
    elif args.model in ["DiffuSeq", "diffuseq", "diffusion"]:
        reaction_model = DiffuSeq()
    else:
        raise NotImplementedError("The model does not yet exist.")

    # set pythonpath
    set_pythonpath(path=os.getcwd())

    # change working dir
    os.chdir(reaction_model.name)

    # instantiate the pipeline with the model
    pipeline = BenchmarkPipeline(model=reaction_model)

    # call pipeline function based on args
    pipeline.run_mode_from_args(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark parser')

    prepare_parser(parser)
    parser.add_argument("--model", "-model", "--m", "-m", type=str, default="ONMT", help="Model to use for benchmarking")
    args = parser.parse_args()

    main(args)
