import argparse
import os
from utils import set_pythonpath, prepare_parser
from g2s_model import G2S
from onmt_model import OpenNMT
from benchmark_models import BenchmarkPipeline


def main(args):

    # instantiate model depending on args
    if args.model in ["Graph2SMILES", 'G2S', 'g2s', 'graph2smiles']:
        reaction_model = G2S()
    elif args.model in ["OpenNMT", 'ONMT', 'onmt', 'opennmt']:
        reaction_model = OpenNMT()
    else:
        raise NotImplementedError("The model does not yet exist.")

    set_pythonpath(path=os.getcwd())

    os.chdir(reaction_model.name)

    pipeline = BenchmarkPipeline(model=reaction_model)

    pipeline.run_mode_from_args(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark parser')

    prepare_parser(parser)
    parser.add_argument("--model", "-model", "--m", "-m", type=str, default="ONMT", help="Model to use for benchmarking")
    args = parser.parse_args()

    main(args)
