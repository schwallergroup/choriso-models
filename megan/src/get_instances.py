"""
Functions for loading configured model, dataset and featurizer instances
"""

import gin
from megan.src import config
from megan.src.datasets import Dataset
from megan.src.feat import ReactionFeaturizer
from megan.src.split import DatasetSplit


@gin.configurable()
def get_dataset(dataset_key: str = gin.REQUIRED) -> Dataset:
    return config.get_dataset(dataset_key)


@gin.configurable()
def get_featurizer(featurizer_key: str = gin.REQUIRED) -> ReactionFeaturizer:
    return config.get_featurizer(featurizer_key)


@gin.configurable()
def get_split(split_key: str = gin.REQUIRED) -> DatasetSplit:
    return config.get_split(split_key)
