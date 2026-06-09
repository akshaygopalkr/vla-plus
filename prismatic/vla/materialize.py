"""
materialize.py

Factory class for initializing Open-X RLDS-backed datasets, given specified data mixture parameters; provides and
exports individual functions for clear control flow.
"""

from pathlib import Path
from typing import Tuple, Type, List

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import EpisodicRLDSDataset, RLDSBatchTransform, RLDSDataset, CotrainingDataset, CotrainingBatchTransform
from prismatic.vla.datasets.rlds.oxe import COTRAINING_NAMED_MIXTURES


def get_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    predict_stop_token: bool = True,
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    episodic: bool = False,
    image_aug: bool = False,
) -> Tuple[Dataset, ActionTokenizer, PaddedCollatorForActionPrediction]:
    """Initialize RLDS Dataset (wraps TFDS), ActionTokenizer, and initialize transform/collation functions."""
    action_tokenizer = ActionTokenizer(tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer, tokenizer, image_transform, prompt_builder_fn, predict_stop_token=predict_stop_token
    )
    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side=padding_side
    )

    # Build RLDS Iterable Dataset
    cls = RLDSDataset if not episodic else EpisodicRLDSDataset
    dataset = cls(
        data_root_dir,
        data_mix,
        batch_transform,
        resize_resolution=default_image_resolution[1:],
        shuffle_buffer_size=shuffle_buffer_size,
        train=train,
        image_aug=image_aug,
    )

    return dataset, action_tokenizer, collator

def get_cotraining_datasets(
    data_root_dir: Path,
    data_mix: str,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    predict_stop_token: bool = True,
    shuffle_buffer_size: int = 100_000,
) -> List[Dataset]:
    """Initialize RLDS Dataset (wraps TFDS), ActionTokenizer, and initialize transform/collation functions."""
    
    cotraining_datasets = []
    
    if data_mix not in COTRAINING_NAMED_MIXTURES:
        raise ValueError(f"Invalid cotraining data mixture: {data_mix}")
    
    datasets = COTRAINING_NAMED_MIXTURES[data_mix]
    
    for dataset_name in datasets:
    
    
        batch_transform = CotrainingBatchTransform(
            tokenizer, image_transform, prompt_builder_fn, 
            predict_stop_token=predict_stop_token, dataset_name=dataset_name
        )
    
        # Build RLDS Iterable Dataset
        cls = CotrainingDataset
        dataset = cls(
            data_root_dir,
            dataset_name,
            batch_transform,
            shuffle_buffer_size=shuffle_buffer_size
        )
        
        cotraining_datasets.append(dataset)

    return cotraining_datasets, datasets
