"""
 RNN tools
"""
import torch.nn as nn
import Graph2SMILES.onmt_v1_2_0.models


def rnn_factory(rnn_type, **kwargs):
    """ rnn factory, Use pytorch version when available. """
    no_pack_padded_seq = False
    if rnn_type == "SRU":
        # SRU doesn't support PackedSequence.
        no_pack_padded_seq = True
        rnn = Graph2SMILES.onmt_v1_2_0.models.sru.SRU(**kwargs)
    else:
        rnn = getattr(nn, rnn_type)(**kwargs)
    return rnn, no_pack_padded_seq
