import argparse
from model_args import ReactionModelArgs
from g2s_model import G2SArgs
from onmt_model import OpenNMTArgs
from hf_model import HuggingFaceArgs


def get_base_parsers():
    # Define the main parser
    parser = argparse.ArgumentParser(description='Conditional arguments example')

    # Define the subparsers for each model
    subparsers = parser.add_subparsers(title='Models', dest='model')

    parser_g2s = subparsers.add_parser('Graph2SMILES', aliases=['G2S', 'g2s', 'graph2smiles'], help='Graph2SMILES model')
    parser_onmt = subparsers.add_parser('OpenNMT', aliases=['ONMT', 'onmt', 'opennmt'], help='OpenNMT model')
    parser_hf = subparsers.add_parser('HuggingFace', aliases=['HF', 'hf', 'huggingface', 'Huggingface'],
                                      help='Huggingface model')

    return parser, {"G2S": {"base_parser": parser_g2s,
                            "args_class": G2SArgs()},
                    "ONMT": {"base_parser": parser_onmt,
                             "args_class": OpenNMTArgs()},
                    "HF": {"base_parser": parser_hf,
                           "args_class": HuggingFaceArgs()}}


def add_mode_subparser(model_parser):
    mode_subparser = model_parser.add_subparsers(title="Run mode")

    train_mode_parser = mode_subparser.add_parser('train', aliases=['t'], help='Training mode')
    predict_mode_parser = mode_subparser.add_parser('predict', aliases=['p', 'pred'], help='Prediction mode')

    return train_mode_parser, predict_mode_parser


def build_parser():

    parser, parser_dict = get_base_parsers()

    for model in parser_dict:
        model_base_parser = parser_dict[model]["base_parser"]
        model_args = parser_dict[model]["args_class"]

        train_parser, predict_parser = add_mode_subparser(model_base_parser)

        train_parser.add_parser('train_args', help='Training args', parser_class=model_args.training_args())
        predict_parser.add_parser('pred_args', help='Prediction args', parser_class=model_args.training_args())

        parser_dict[model]["train_parser"] = train_parser
        parser_dict[model]["predict_parser"] = predict_parser

    return parser


def main(parser):
    args = parser.parse_args()
    # instantiate model depending on args

    """
    if args.model = G2S:
        reaction_model = G2S()
    elif args.model = ONMT:
        reaction_model = OpenNMT()
    elif args.model = HFTransformer:
        reaction_model = HuggingFaceTransformer()
    else:
        raise NotImplementedError()
    """

    # instantiate the pipeline with the model
    """
    pipeline = BenchmarkPipeline(model=reaction_model)
    """

    # call pipeline function based on args
    """
    
    """


if __name__ == "__main__":
    argparser = build_parser()
    main(argparser)
