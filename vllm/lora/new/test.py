"""
This script is mainly used to tests various hidden_sizes. We have collected the
hidden_sizes included in the LoRA models currently supported by vLLM. It tests
whether the corresponding Triton kernel can run normally when tensor parallelism
is set to [1, 2, 4, 8, 16, 32, 64].
"""

import pytest
import torch
from typing import Tuple

"""
Based on:
Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023). 
Punica: Multi-Tenant LoRA Serving. 
https://arxiv.org/abs/2310.18547
"""

import torch
import triton
import triton.language as tl
import numpy as np
import random


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)






@triton.jit
def _sgmv_expand_slice_kernel(
    input_ptr,
    lora_ptr_adrs,
    out_ptr,
    N,
    K,
    b_seq_start_loc,
    seq_lens,
    lora_indices,
    slice_start_loc,
    xm_stride,
    xk_stride,  # 1
    l0_stride,  # hidden_size*max_rank
    lora_k_stride,
    lora_n_stride,
    cm_stride,
    cn_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    CAST_TYPE: tl.constexpr,
):
    """

    Similar to the 'sgmv_expand' operator, but with an added parameter
    'slice_offset'. The reason for not reusing the 'sgmv_expand' operator
    might be that in the future, we could implement a fusion operator to
    achieve the current functionality instead of having to call it multiple
    times.
    """
    pid = tl.program_id(axis=0)
    cur_batch = tl.program_id(axis=1)
    slice_id = tl.program_id(axis=2)
    cta_n_num = tl.cdiv(N, BLOCK_N)
    pid_m = pid // cta_n_num
    pid_n = pid % cta_n_num
    M = tl.load(seq_lens + cur_batch)
    if pid_m * BLOCK_M > M:
        return
    lora_index = tl.load(lora_indices + cur_batch)
    if lora_index == -1:
        return

    cur_seq_start = tl.load(b_seq_start_loc + cur_batch)
    offset_m = tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
    offset_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    offset_k = tl.arange(0, BLOCK_K)
    ram = tl.max_contiguous(tl.multiple_of(offset_m % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(offset_n % N, BLOCK_N), BLOCK_N)
    # 取每个slice对应的lora指针地址
    lora_ptr = tl.load(lora_ptr_adrs + slice_id).to(
        tl.pointer_type(out_ptr.dtype.element_ty)
    )
    a_ptr = (
        input_ptr
        + cur_seq_start * xm_stride
        + ram[:, None] * xm_stride
        + offset_k[None, :] * xk_stride,
    )
    b_ptr = (
        lora_ptr
        + l0_stride * lora_index
        + offset_k[:, None] * lora_n_stride
        + rbn[None, :] * lora_k_stride
    )
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            tiled_a = tl.load(a_ptr)
            tiled_b = tl.load(b_ptr)
        else:
            tiled_a = tl.load(a_ptr, mask=offset_k[None, :] < K - k * BLOCK_K, other=0)
            tiled_b = tl.load(b_ptr, mask=offset_k[:, None] < K - k * BLOCK_K, other=0)
        if CAST_TYPE:
            tiled_a = tiled_a.to(lora_ptr.dtype.element_ty)
        accumulator += tl.dot(
            tiled_a,
            tiled_b,
        )
        a_ptr += BLOCK_K * xk_stride
        b_ptr += BLOCK_K * lora_n_stride

    tiled_c = accumulator.to(lora_ptr.dtype.element_ty)
    # 获取每个slice的偏移地址
    if slice_start_loc is not None:
        cur_slice_start = tl.load(slice_start_loc + slice_id)
    else:
        cur_slice_start=0
    offset_cm = cur_seq_start + tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
    offset_cn = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N + cur_slice_start
    c_ptr = out_ptr + offset_cm[:, None] * cm_stride + offset_cn[None, :] * cn_stride
    c_mask = (offset_cm[:, None] < (cur_seq_start + M)) & (
        offset_cn[None, :] < (cur_slice_start + N)
    )
    if ADD_INPUTS:
        tiled_out = tl.load(c_ptr, mask=c_mask)
        tiled_c += tiled_out
    tl.store(c_ptr, tiled_c, mask=c_mask)


@torch.inference_mode()
def _sgmv_expand_slice(
    inputs: torch.Tensor,
    lora_b_stacked: Tuple[torch.Tensor,...],
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    slice_start_loc: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    add_inputs: bool = False,
) -> None:


    assert inputs.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert lora_b_stacked[0].dtype in [
        torch.float16,
        torch.bfloat16,
    ]
    assert inputs.size(0) == token_nums

    assert b_seq_start_loc.size(0) == batches
    assert lora_indices_tensor.size(0) == batches
    assert inputs.is_contiguous()
    assert output_tensor.is_contiguous()

    tensor_ptrs = []
    for lora_b_weight in lora_b_stacked:
        if lora_b_weight.ndim == 4:  # shape:(lora_num,1,size,rank)
            assert lora_b_weight.size(1) == 1
            lora_b_weight = lora_b_weight.squeeze(dim=1)
        else:
            assert lora_b_weight.ndim == 3  # shape:(lora_num,size,rank)
        assert lora_b_weight.is_contiguous()
        tensor_ptrs.append(lora_b_weight.data_ptr())

    # note these are device tensors
    lora_ptr_tensor = torch.tensor(tensor_ptrs, device=b_seq_start_loc.device)

    # TODO tuning this config
    N, K = lora_b_weight.shape[-2:]  # K= rank,N=hidden_size

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = 16
    EVEN_K = K % BLOCK_K == 0
    ADD_INPUTS = add_inputs
    CAST_TYPE = False
    if inputs.dtype == torch.float32 and lora_b_stacked[0].dtype in [
        torch.float16,
        torch.bfloat16,
    ]:
        CAST_TYPE = True
    grid = (
        triton.cdiv(max_seq_length, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        batches,
        len(lora_ptr_tensor),
    )
    _sgmv_expand_slice_kernel[grid](
        inputs,
        lora_ptr_tensor,
        output_tensor,
        N,
        K,
        b_seq_start_loc,
        seq_len_tensor,
        lora_indices_tensor,
        slice_start_loc,
        inputs.stride(0),
        inputs.stride(1),
        lora_b_weight.stride(0),
        lora_b_weight.stride(1),
        lora_b_weight.stride(2),
        output_tensor.stride(0),
        output_tensor.stride(1),
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        ADD_INPUTS,
        CAST_TYPE,
    )
    return


if __name__ == "__main__":
    from utils import (
        generate_data,
        generate_data_for_expand_nslices,
        ref_torch_groupgemm,
    )

else:
    from .utils import (
        generate_data,
        generate_data_for_expand_nslices,
        ref_torch_groupgemm,
    )


HIDDEN_SIZES = [
    128,
    256,
    512,
    896,
    1024,
    1152,
    1216,
    1280,
    1536,
    1664,
    2048,
    2240,
    2304,
    2368,
    2432,
    2560,
    2752,
    3072,
    3328,
    3456,
    3584,
    3712,
    4096,
    4480,
    4608,
    4736,
    4864,
    5120,
    5504,
    5632,
    5888,
    6144,
    6400,
    6848,
    6912,
    7168,
    7424,
    8192,
    8960,
    9216,
    9472,
    10240,
    11008,
    11264,
    13824,
    14336,
    14784,
    14848,
    15360,
    18944,
    22016,
    22528,
    24576,
    27392,
    27648,
    29568,
    29696,
    32000,
    32256,
    32512,
    32768,
    33024,
    36864,
    43264,
    49152,
    49408,
    60544,
    60672,
    64000,
    64256,
    102400,
    102656,
    128000,
    128256,
]
# The size of TP
divisibility = [1, 2, 8, 16, 64]

all_hidden_size = []
for div in divisibility:
    for hidden_size in HIDDEN_SIZES:
        all_hidden_size.append(hidden_size // div)

HIDDEN_SIZES = list(set(all_hidden_size))

BATCHES = [4]
NUM_LORA = [4]
DTYPES = [torch.float16, torch.bfloat16]
MAX_RANKS = [32]
SCALES = [0.5]
SEED = [0]
CUDA_DEVICES = [f"cuda:{0}"]


def assert_close(a, b):
    rtol, atol = {
        torch.float16: (6e-2, 6e-2),
        torch.bfloat16: (6e-2, 6e-2),
        torch.float32: (1e-2, 1e-2),
    }[a.dtype]
    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


def test_punica_expand_nslices_fused(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    nslices: int,
    dtype: torch.dtype,
    op_type: str,
    seed: int,
    device: str,
):
    seed_everything(seed)

    seq_length = 128 if op_type == "sgmv" else 1
    (
        inputs_tensor,
        lora_weights_lst,
        our_outputs,
        ref_outputs,
        b_seq_start_loc,
        lora_indices_tensor,
        seq_len_tensor,
        indices,
    ) = generate_data_for_expand_nslices(
        batches,
        hidden_size,
        num_loras,
        rank,
        seq_length,
        dtype,
        nslices,
        device,
    )
    max_seq_length = seq_len_tensor.max()
    token_nums = seq_len_tensor.sum().item()
    if isinstance(max_seq_length, tuple):
        max_seq_length = max_seq_length[0].item()
    else:
        max_seq_length = max_seq_length.item()
    slice_offset = 0

    slice_offset_lst = []
    for index in range(nslices):
        lora_weights = lora_weights_lst[index]
        ref_torch_groupgemm(
            ref_outputs[:, slice_offset : slice_offset + hidden_size],
            inputs_tensor,
            lora_weights,
            lora_indices_tensor,
            seq_len_tensor,
            batches,
            1.0,
            op_type="expand",
        )
        slice_offset_lst.append(slice_offset)
        slice_offset += hidden_size
    slice_start_loc = torch.tensor(slice_offset_lst).cuda()
    if op_type == "sgmv":
        _sgmv_expand_slice(
            inputs_tensor,
            lora_weights_lst,
            our_outputs,
            b_seq_start_loc,
            seq_len_tensor,
            lora_indices_tensor,
            slice_start_loc,
            batches,
            max_seq_length,
            token_nums,
            add_inputs=True,
        )
    assert_close(our_outputs, ref_outputs)


if __name__ == "__main__":
    from itertools import product

    for ele in product(
        BATCHES,
        NUM_LORA,
        MAX_RANKS,
        HIDDEN_SIZES,
        [2, 3],
        DTYPES,
        ["sgmv"],
        SEED,
        CUDA_DEVICES,
    ):
        try:
            test_punica_expand_nslices_fused(*ele)
            print(f"{ele} passed")
        except Exception as error:
            raise error
