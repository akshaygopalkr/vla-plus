from typing import Optional, Sequence, Type

import torch
from torch import nn as nn
from transformers import GemmaForCausalLM
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial
from typing import Callable
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn import Embedding
# from prismatic.util.nn_utils import LMHead, EmbeddingProjector

from prismatic.models.backbones.llm.base_llm import HFCausalLLMBackbone
from prismatic.models.backbones.llm.prompting import (
    LLaMa2ChatPromptBuilder,
    PromptBuilder,
    PurePromptBuilder,
    VicunaV15ChatPromptBuilder,
)

# Registry =>> Support Gemma Models (from HF Transformers)
# fmt: off
GEMMA_MODELS = {
    "gemma-2b": {
        "llm_family": "gemma", "llm_cls": GemmaForCausalLM, "hf_hub_path": "agopalkr/gemma-2b"
    },

}


# fmt: on


class GemmaLLMBackbone(HFCausalLLMBackbone):
    def __init__(
            self,
            llm_backbone_id: str,
            llm_max_length: int = 2048,
            hf_token: Optional[str] = None,
            inference_mode: bool = False,
            use_flash_attention_2: bool = True,
    ) -> None:
        super().__init__(
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            use_flash_attention_2=use_flash_attention_2,
            **GEMMA_MODELS[llm_backbone_id],
        )

        # [Special Case] LLaMa-2 PAD Token Handling --> for clarity, we add an extra token (and resize)
        self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id

        # Add prompt embeddings to tokenizer
        prompt_embeddings = ["<VLM_PROMPT>", "<VLA_PROMPT>", "<VIS_TRACE_PROMPT>"]
        self.tokenizer.add_tokens(prompt_embeddings)
        self.llm.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        return PurePromptBuilder

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return GemmaDecoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        """LLaMa-2 was trained in BF16; see https://huggingface.co/docs/transformers/main/model_doc/llama2."""
        return torch.bfloat16

    @property
    def last_layer_finetune_modules(self) -> Sequence[nn.Module]:
        return (self.llm.model.embed_tokens, self.llm.model.layers[-1], self.llm.lm_head)