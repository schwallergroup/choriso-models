"""  Attention and normalization modules  """
from Graph2SMILES.onmt_v1_2_0.modules.util_class import Elementwise
from Graph2SMILES.onmt_v1_2_0.modules.gate import context_gate_factory, ContextGate
from Graph2SMILES.onmt_v1_2_0.modules.global_attention import GlobalAttention
from Graph2SMILES.onmt_v1_2_0.modules.conv_multi_step_attention import ConvMultiStepAttention
from Graph2SMILES.onmt_v1_2_0.modules.copy_generator import CopyGenerator, CopyGeneratorLoss, \
    CopyGeneratorLossCompute
from Graph2SMILES.onmt_v1_2_0.modules.multi_headed_attn import MultiHeadedAttention
from Graph2SMILES.onmt_v1_2_0.modules.embeddings import Embeddings, PositionalEncoding, \
    VecEmbedding
from Graph2SMILES.onmt_v1_2_0.modules.weight_norm import WeightNormConv2d
from Graph2SMILES.onmt_v1_2_0.modules.average_attn import AverageAttention

import Graph2SMILES.onmt_v1_2_0.modules.source_noise # noqa

__all__ = ["Elementwise", "context_gate_factory", "ContextGate",
           "GlobalAttention", "ConvMultiStepAttention", "CopyGenerator",
           "CopyGeneratorLoss", "CopyGeneratorLossCompute",
           "MultiHeadedAttention", "Embeddings", "PositionalEncoding",
           "WeightNormConv2d", "AverageAttention", "VecEmbedding"]
