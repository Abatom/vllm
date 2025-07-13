# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine import (
    P2pNcclEngine)
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import MLACommonMetadata

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.request import Request

logger = init_logger(__name__)


def inject_kv_into_layer(
    attn_metadata: "AttentionMetadata",
    dst_kv_cache_layer: torch.Tensor,
    src_kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    request_id: str,
) -> None:
    """Inject the KV cache into the layer.

    Args:
        dst_kv_cache_layer (torch.Tensor): the destination KV cache
            layer. In shape [2, num_pages, page_size, xxx] if not
            using MLA, [num_pages, page_size, xxx] otherwise.
        src_kv_cache (torch.Tensor): the source KV cache. In shape
            [2, num_tokens, xxx] if not using MLA, [num_tokens, xxx]
            otherwise.
        slot_mapping (torch.Tensor): the slot mapping. In shape
            [num_tokens].
        request_id (str): request id for log
    """
    dst_kv_cache_layer_shape = dst_kv_cache_layer.shape
    if isinstance(attn_metadata, MLACommonMetadata):
        num_pages = dst_kv_cache_layer_shape[0]
        page_size = dst_kv_cache_layer_shape[1]
        dst_kv_cache_layer = dst_kv_cache_layer.reshape(
            num_pages * page_size, -1)
        self.check_tensors_except_dim(dst_kv_cache_layer, src_kv_cache, 0)
        num_token = src_kv_cache.shape[0]
        if len(slot_mapping) == num_token:
            dst_kv_cache_layer[slot_mapping, ...] = src_kv_cache
        else:
            dst_kv_cache_layer[slot_mapping[:num_token], ...] = src_kv_cache
            logger.warning(
                "🚧src_kv_cache does not match, num_slot:%d, "
                "num_token:%d, request_id:%s", len(slot_mapping),
                num_token, request_id)

        dst_kv_cache_layer.reshape(dst_kv_cache_layer_shape)
        return

    num_pages = dst_kv_cache_layer_shape[1]
    page_size = dst_kv_cache_layer_shape[2]
    dst_kv_cache_layer = dst_kv_cache_layer.reshape(
        2, num_pages * page_size, -1)
    self.check_tensors_except_dim(dst_kv_cache_layer, src_kv_cache, 1)
    num_token = src_kv_cache.shape[1]
    if len(slot_mapping) == num_token:
        dst_kv_cache_layer[:, slot_mapping, ...] = src_kv_cache
    else:
        dst_kv_cache_layer[:, slot_mapping[:num_token], ...] = src_kv_cache
        logger.warning(
            "🚧src_kv_cache does not match, num_slot:%d, "
            "num_token:%d, request_id:%s", len(slot_mapping),
            num_token, request_id)

    dst_kv_cache_layer.reshape(dst_kv_cache_layer_shape)


def extract_kv_from_layer(
    attn_metadata: "AttentionMetadata",
    layer: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> torch.Tensor:
    """Extract the KV cache from the layer.

    Assume the shape of the layer is (2, num_pages, page_size, xxx)
    if MLA is not used, and (num_pages, page_size, xxx) otherwise.
    """
    if isinstance(attn_metadata, MLACommonMetadata):
        num_pages, page_size = layer.shape[0], layer.shape[1]
        return layer.reshape(num_pages * page_size, -1)[slot_mapping, ...]

    num_pages, page_size = layer.shape[1], layer.shape[2]
    return layer.reshape(2, num_pages * page_size, -1)[:, slot_mapping, ...]