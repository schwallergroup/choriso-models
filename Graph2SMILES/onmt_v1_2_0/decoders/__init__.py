"""Module defining decoders."""
from Graph2SMILES.onmt_v1_2_0.decoders.decoder import DecoderBase, InputFeedRNNDecoder, \
    StdRNNDecoder
from Graph2SMILES.onmt_v1_2_0.decoders.transformer import TransformerDecoder
from Graph2SMILES.onmt_v1_2_0.decoders.cnn_decoder import CNNDecoder


str2dec = {"rnn": StdRNNDecoder, "ifrnn": InputFeedRNNDecoder,
           "cnn": CNNDecoder, "transformer": TransformerDecoder}

__all__ = ["DecoderBase", "TransformerDecoder", "StdRNNDecoder", "CNNDecoder",
           "InputFeedRNNDecoder", "str2dec"]
