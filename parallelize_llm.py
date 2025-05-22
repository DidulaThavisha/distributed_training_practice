# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy

from torchtitan.config_manager import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger


def parallelize_llm(
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    if job_config.activation_checkpoint.mode != "none":
        apply_ac(model, job_config.activation_checkpoint)

    if (
        parallel_dims.dp_shard_enabled or parallel_dims.dp_replicate_enabled
    ):  # apply FSDP or HSDP
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = (["dp_replicate"])
        elif parallel_dims.dp_shard_enabled:
            dp_mesh_dim_names = ("dp",)
        elif parallel_dims.dp_replicate_enabled and parallel_dims.dp_shard_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp")
        else:
            raise ValueError(
                "Either dp_replicate or dp_shard must be enabled for FSDP."
            )
        # Apply FSDP to the model

        apply_fsdp(
            model,
            world_mesh[tuple(dp_mesh_dim_names)],
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            cpu_offload=job_config.training.enable_cpu_offload,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

    return model


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    cpu_offload: bool = False,
):
    """
    Apply data parallelism (via FSDP2) to the model.

    Args:
        model (nn.Module): The model to apply data parallelism to.
        dp_mesh (DeviceMesh): The device mesh to use for data parallelism.
        param_dtype (torch.dtype): The data type to use for model parameters.
        reduce_dtype (torch.dtype): The data type to use for reduction operations.
        cpu_offload (bool): Whether to offload model parameters to CPU. Defaults to False.
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    linear_layers = [
        model.token_embed,
        model.position_embed,
    ]
    for layer in linear_layers:
        fully_shard(layer, **fsdp_config)

    # for block in model.double_blocks:
    #     fully_shard(
    #         block,
    #         **fsdp_config,
    #     )

    for block in model.attn_blocks:
        fully_shard(
            block,
            **fsdp_config,
        )
    # apply FSDP to last layer. Set reshard_after_forward=False for last layer to avoid gather right after reshard
    fully_shard(model.lm_head, **fsdp_config, reshard_after_forward=False)

    # Wrap all the rest of model
    fully_shard(model, **fsdp_config)


def apply_ac(model: nn.Module, ac_config):
    """Apply activation checkpointing to the model."""

    for layer_id, block_module in enumerate(model.attn_blocks): # Use enumerate if blocks is a list/sequential
        # If model.blocks.named_children() is preferred and works for your structure:
        # for layer_id, block_module in model.blocks.named_children():
        checkpointed_block = ptd_checkpoint_wrapper(block_module, preserve_rng_state=False)
        # If model.blocks is nn.Sequential, you might need to replace the module by index
        if isinstance(model.attn_blocks, nn.Sequential):
            model.attn_blocks[int(layer_id)] = checkpointed_block
        elif isinstance(model.attn_blocks, nn.ModuleList):
                model.attn_blocks[int(layer_id)] = checkpointed_block

    logger.info(f"Applied {ac_config.mode} activation checkpointing to the model")
