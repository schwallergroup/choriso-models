import argparse
from model_args import ReactionModelArgs
from g2s_model import G2SArgs, G2S
from onmt_model import OpenNMTArgs, OpenNMT
from hf_model import HuggingFaceArgs, HuggingFaceTransformer
from neuralsym_model import NeuralsymArgs, Neuralsym
from benchmark_models import BenchmarkPipeline


def get_base_parsers():
    # Define the main parser
    parser = argparse.ArgumentParser(description='Conditional arguments example')

    # Define the subparsers for each model
    subparsers = parser.add_subparsers(title='Models', dest='model', required=True)

    parser_g2s = subparsers.add_parser('Graph2SMILES', aliases=['G2S', 'g2s', 'graph2smiles'], help='Graph2SMILES model')
    parser_onmt = subparsers.add_parser('OpenNMT', aliases=['ONMT', 'onmt', 'opennmt'], help='OpenNMT model')
    parser_hf = subparsers.add_parser('HuggingFace', aliases=['HF', 'hf', 'huggingface', 'Huggingface'],
                                      help='Huggingface model')
    parser_ns = subparsers.add_parser('Neuralsym', aliases=['NS', 'ns', 'neuralsym', 'NeuralSym'], help='Neuralsym model')

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

    train_mode_parser = mode_subparser.add_parser('train', aliases=['t'], help='Training mode') # action='store_true')
    predict_mode_parser = mode_subparser.add_parser('predict', aliases=['p', 'pred'], help='Prediction mode') # action='store_true')

    return train_mode_parser, predict_mode_parser


def build_parser():

    parser, parser_dict = get_base_parsers()

    for model in parser_dict:
        model_base_parser = parser_dict[model]["base_parser"]
        model_args = parser_dict[model]["args_class"]

        train_parser, predict_parser = add_mode_subparser(model_base_parser)

        # train_subparser = train_parser.add_subparsers(title='train_args', help='Training args')
        for train_arg in model_args.training_args()._actions:
            try:
                train_parser._add_action(train_arg)
            except:
                continue

        # predict_subparser = predict_parser.add_subparsers(title='pred_args', help='Prediction args')
        for pred_arg in model_args.predict_args()._actions:
            try:
                predict_parser._add_action(pred_arg)
            except:
                continue

        parser_dict[model]["train_parser"] = train_parser
        parser_dict[model]["predict_parser"] = predict_parser

    return parser


def main(parser):
    args = parser.parse_args()
    # instantiate model depending on args
    print(args)
    if args.model == "Graph2SMILES":
        reaction_model = G2S()
    elif args.model == "OpenNMT":
        reaction_model = OpenNMT()
    elif args.model == "HuggingFace":
        reaction_model = HuggingFaceTransformer()
    else:
        raise NotImplementedError("The model does not yet exist.")

    # instantiate the pipeline with the model

    pipeline = BenchmarkPipeline(model=reaction_model)

    # call pipeline function based on args
    if args.train:
        pipeline.run_train_pipeline()
    elif args.predict:
        pipeline.predict(dataset=args.dataset)


if __name__ == "__main__":
    argparser = build_parser()
    main(argparser)
