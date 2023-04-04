"""Module defining encoders."""
from Graph2SMILES.onmt_v1_2_0.encoders.encoder import EncoderBase
from Graph2SMILES.onmt_v1_2_0.encoders.transformer import TransformerEncoder
from Graph2SMILES.onmt_v1_2_0.encoders.ggnn_encoder import GGNNEncoder
from Graph2SMILES.onmt_v1_2_0.encoders.rnn_encoder import RNNEncoder
from Graph2SMILES.onmt_v1_2_0.encoders.cnn_encoder import CNNEncoder
from Graph2SMILES.onmt_v1_2_0.encoders.mean_encoder import MeanEncoder
from Graph2SMILES.onmt_v1_2_0.encoders.audio_encoder import AudioEncoder
from Graph2SMILES.onmt_v1_2_0.encoders.image_encoder import ImageEncoder


str2enc = {"ggnn": GGNNEncoder, "rnn": RNNEncoder, "brnn": RNNEncoder,
           "cnn": CNNEncoder, "transformer": TransformerEncoder,
           "img": ImageEncoder, "audio": AudioEncoder, "mean": MeanEncoder}

__all__ = ["EncoderBase", "TransformerEncoder", "RNNEncoder", "CNNEncoder",
           "MeanEncoder", "str2enc"]
