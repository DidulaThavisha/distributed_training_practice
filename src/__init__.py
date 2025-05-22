# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.loss import build_cross_entropy_loss
from data_loader.data_loader import build_llm_dataloader

from parallelize_llm import parallelize_llm
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from src.models.transformer import Transformer, TransformerArgs

__all__ = [
    "Transformer"
    "TransformerArgs",
]

transformer_configs = {
    "llm_small": TransformerArgs(
        n_head=16,
        n_embed=768,
        context_length=1024,
        vocab_size=50265,
        N_BLOCKS=16,
    ),

}


register_train_spec(
    TrainSpec(
        name="llm",
        cls=Transformer,
        config=transformer_configs,
        parallelize_fn=parallelize_llm,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_llm_dataloader,
        build_tokenizer_fn=None,
        build_loss_fn=build_cross_entropy_loss,
    )
)
