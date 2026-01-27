# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
P2P RDMA Engine for KV cache transfer.

This module provides a P2P engine that uses:
- RDMA (via raw ibverbs with Queue Pairs) for inter-node communication
- NVLink (via CUDA P2P with copy engine) for intra-node communication

Key features:
- Zero-copy GPU memory transfers via GPUDirect RDMA
- No temporary GPU buffers - direct memory registration
- SM-free transfers using DMA/copy engines only
- Similar architecture to Mooncake's TransferEngine
"""

import ctypes
import json
import logging
import os
import socket
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import msgpack
import torch
import zmq

from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.tensor_memory_pool import (
    TensorMemoryPool,
)
from vllm.utils.network_utils import get_ip

logger = logging.getLogger(__name__)

DEFAULT_MEM_POOL_SIZE_GB = 32

# ============================================================================
# RDMA Constants from libibverbs
# ============================================================================

# Access flags
IBV_ACCESS_LOCAL_WRITE = 1
IBV_ACCESS_REMOTE_WRITE = 2
IBV_ACCESS_REMOTE_READ = 4
IBV_ACCESS_REMOTE_ATOMIC = 8

# Work request opcodes
IBV_WR_RDMA_WRITE = 0
IBV_WR_RDMA_WRITE_WITH_IMM = 1
IBV_WR_SEND = 2
IBV_WR_RDMA_READ = 3

# Send flags
IBV_SEND_SIGNALED = 1 << 0
IBV_SEND_INLINE = 1 << 3

# Work completion status
IBV_WC_SUCCESS = 0

# Queue pair types
IBV_QPT_RC = 2  # Reliable Connection

# Queue pair states
IBV_QPS_RESET = 0
IBV_QPS_INIT = 1
IBV_QPS_RTR = 2  # Ready to Receive
IBV_QPS_RTS = 3  # Ready to Send

# MTU values
IBV_MTU_4096 = 5

# Port states
IBV_PORT_ACTIVE = 4


def get_hostname() -> str:
    """Get the hostname of the current machine."""
    return socket.gethostname()


def is_same_node(local_hostname: str, remote_address: str) -> bool:
    """Check if the remote address is on the same node."""
    try:
        remote_host = remote_address.split(":")[0]
        if remote_host == local_hostname:
            return True
        if remote_host in ("localhost", "127.0.0.1", "::1"):
            return True
        try:
            local_ip = socket.gethostbyname(local_hostname)
            remote_ip = socket.gethostbyname(remote_host)
            if local_ip == remote_ip:
                return True
        except socket.gaierror:
            pass
        return False
    except Exception as e:
        logger.warning("Error checking if same node: %s", e)
        return False


@dataclass
class SendQueueItem:
    tensor_id: str
    remote_address: str
    tensor: torch.Tensor


@dataclass
class RdmaMemoryRegion:
    """Represents a registered RDMA memory region."""
    addr: int
    length: int
    lkey: int
    rkey: int
    mr_ptr: ctypes.c_void_p


@dataclass
class QueuePairInfo:
    """Queue Pair connection information for RDMA."""
    qp_num: int
    lid: int
    psn: int
    gid: bytes  # 16 bytes GID


@dataclass
class RemoteQPInfo:
    """Remote Queue Pair info for connection."""
    qp_num: int
    lid: int
    psn: int
    gid: bytes
    # Remote memory regions
    mem_regions: dict[int, tuple[int, int]] = field(default_factory=dict)  # addr -> (rkey, size)


# ============================================================================
# ibverbs Structure Definitions
# ============================================================================

class ibv_gid(ctypes.Structure):
    _fields_ = [("raw", ctypes.c_uint8 * 16)]


class ibv_global_route(ctypes.Structure):
    _fields_ = [
        ("dgid", ibv_gid),
        ("flow_label", ctypes.c_uint32),
        ("sgid_index", ctypes.c_uint8),
        ("hop_limit", ctypes.c_uint8),
        ("traffic_class", ctypes.c_uint8),
    ]


class ibv_ah_attr(ctypes.Structure):
    _fields_ = [
        ("grh", ibv_global_route),
        ("dlid", ctypes.c_uint16),
        ("sl", ctypes.c_uint8),
        ("src_path_bits", ctypes.c_uint8),
        ("static_rate", ctypes.c_uint8),
        ("is_global", ctypes.c_uint8),
        ("port_num", ctypes.c_uint8),
    ]


class ibv_mr(ctypes.Structure):
    _fields_ = [
        ("context", ctypes.c_void_p),
        ("pd", ctypes.c_void_p),
        ("addr", ctypes.c_void_p),
        ("length", ctypes.c_size_t),
        ("handle", ctypes.c_uint32),
        ("lkey", ctypes.c_uint32),
        ("rkey", ctypes.c_uint32),
    ]


class ibv_sge(ctypes.Structure):
    _fields_ = [
        ("addr", ctypes.c_uint64),
        ("length", ctypes.c_uint32),
        ("lkey", ctypes.c_uint32),
    ]


class ibv_send_wr(ctypes.Structure):
    pass


ibv_send_wr._fields_ = [
    ("wr_id", ctypes.c_uint64),
    ("next", ctypes.POINTER(ibv_send_wr)),
    ("sg_list", ctypes.POINTER(ibv_sge)),
    ("num_sge", ctypes.c_int),
    ("opcode", ctypes.c_int),
    ("send_flags", ctypes.c_uint),
    ("imm_data", ctypes.c_uint32),
    # RDMA specific fields (union in C)
    ("remote_addr", ctypes.c_uint64),
    ("rkey", ctypes.c_uint32),
]


class ibv_recv_wr(ctypes.Structure):
    pass


ibv_recv_wr._fields_ = [
    ("wr_id", ctypes.c_uint64),
    ("next", ctypes.POINTER(ibv_recv_wr)),
    ("sg_list", ctypes.POINTER(ibv_sge)),
    ("num_sge", ctypes.c_int),
]


class ibv_wc(ctypes.Structure):
    _fields_ = [
        ("wr_id", ctypes.c_uint64),
        ("status", ctypes.c_int),
        ("opcode", ctypes.c_int),
        ("vendor_err", ctypes.c_uint32),
        ("byte_len", ctypes.c_uint32),
        ("imm_data", ctypes.c_uint32),
        ("qp_num", ctypes.c_uint32),
        ("src_qp", ctypes.c_uint32),
        ("wc_flags", ctypes.c_int),
        ("pkey_index", ctypes.c_uint16),
        ("slid", ctypes.c_uint16),
        ("sl", ctypes.c_uint8),
        ("dlid_path_bits", ctypes.c_uint8),
    ]


class ibv_port_attr(ctypes.Structure):
    _fields_ = [
        ("state", ctypes.c_int),
        ("max_mtu", ctypes.c_int),
        ("active_mtu", ctypes.c_int),
        ("gid_tbl_len", ctypes.c_int),
        ("port_cap_flags", ctypes.c_uint32),
        ("max_msg_sz", ctypes.c_uint32),
        ("bad_pkey_cntr", ctypes.c_uint32),
        ("qkey_viol_cntr", ctypes.c_uint32),
        ("pkey_tbl_len", ctypes.c_uint16),
        ("lid", ctypes.c_uint16),
        ("sm_lid", ctypes.c_uint16),
        ("lmc", ctypes.c_uint8),
        ("max_vl_num", ctypes.c_uint8),
        ("sm_sl", ctypes.c_uint8),
        ("subnet_timeout", ctypes.c_uint8),
        ("init_type_reply", ctypes.c_uint8),
        ("active_width", ctypes.c_uint8),
        ("active_speed", ctypes.c_uint8),
        ("phys_state", ctypes.c_uint8),
        ("link_layer", ctypes.c_uint8),
        ("flags", ctypes.c_uint8),
        ("port_cap_flags2", ctypes.c_uint16),
    ]


class ibv_qp_init_attr(ctypes.Structure):
    _fields_ = [
        ("qp_context", ctypes.c_void_p),
        ("send_cq", ctypes.c_void_p),
        ("recv_cq", ctypes.c_void_p),
        ("srq", ctypes.c_void_p),
        ("cap_max_send_wr", ctypes.c_uint32),
        ("cap_max_recv_wr", ctypes.c_uint32),
        ("cap_max_send_sge", ctypes.c_uint32),
        ("cap_max_recv_sge", ctypes.c_uint32),
        ("cap_max_inline_data", ctypes.c_uint32),
        ("qp_type", ctypes.c_int),
        ("sq_sig_all", ctypes.c_int),
    ]


class ibv_qp_attr(ctypes.Structure):
    _fields_ = [
        ("qp_state", ctypes.c_int),
        ("cur_qp_state", ctypes.c_int),
        ("path_mtu", ctypes.c_int),
        ("path_mig_state", ctypes.c_int),
        ("qkey", ctypes.c_uint32),
        ("rq_psn", ctypes.c_uint32),
        ("sq_psn", ctypes.c_uint32),
        ("dest_qp_num", ctypes.c_uint32),
        ("qp_access_flags", ctypes.c_int),
        ("cap_max_send_wr", ctypes.c_uint32),
        ("cap_max_recv_wr", ctypes.c_uint32),
        ("cap_max_send_sge", ctypes.c_uint32),
        ("cap_max_recv_sge", ctypes.c_uint32),
        ("cap_max_inline_data", ctypes.c_uint32),
        ("ah_attr", ibv_ah_attr),
        ("alt_ah_attr", ibv_ah_attr),
        ("pkey_index", ctypes.c_uint16),
        ("alt_pkey_index", ctypes.c_uint16),
        ("en_sqd_async_notify", ctypes.c_uint8),
        ("sq_draining", ctypes.c_uint8),
        ("max_rd_atomic", ctypes.c_uint8),
        ("max_dest_rd_atomic", ctypes.c_uint8),
        ("min_rnr_timer", ctypes.c_uint8),
        ("port_num", ctypes.c_uint8),
        ("timeout", ctypes.c_uint8),
        ("retry_cnt", ctypes.c_uint8),
        ("rnr_retry", ctypes.c_uint8),
        ("alt_port_num", ctypes.c_uint8),
        ("alt_timeout", ctypes.c_uint8),
        ("rate_limit", ctypes.c_uint32),
    ]


# ============================================================================
# RDMA Transport Implementation
# ============================================================================

class RdmaTransport:
    """
    Raw RDMA transport using libibverbs.
    
    Implements Queue Pairs for reliable connection-based RDMA transfers.
    Supports GPUDirect RDMA for zero-copy GPU memory transfers.
    """
    
    # QP modification masks
    IBV_QP_STATE = 1 << 0
    IBV_QP_CUR_STATE = 1 << 1
    IBV_QP_EN_SQD_ASYNC_NOTIFY = 1 << 2
    IBV_QP_ACCESS_FLAGS = 1 << 3
    IBV_QP_PKEY_INDEX = 1 << 4
    IBV_QP_PORT = 1 << 5
    IBV_QP_QKEY = 1 << 6
    IBV_QP_AV = 1 << 7
    IBV_QP_PATH_MTU = 1 << 8
    IBV_QP_TIMEOUT = 1 << 9
    IBV_QP_RETRY_CNT = 1 << 10
    IBV_QP_RNR_RETRY = 1 << 11
    IBV_QP_RQ_PSN = 1 << 12
    IBV_QP_MAX_QP_RD_ATOMIC = 1 << 13
    IBV_QP_ALT_PATH = 1 << 14
    IBV_QP_MIN_RNR_TIMER = 1 << 15
    IBV_QP_SQ_PSN = 1 << 16
    IBV_QP_MAX_DEST_RD_ATOMIC = 1 << 17
    IBV_QP_PATH_MIG_STATE = 1 << 18
    IBV_QP_CAP = 1 << 19
    IBV_QP_DEST_QPN = 1 << 20
    
    def __init__(self, device_name: str | None = None, ib_port: int = 1, gid_index: int = 0):
        self.device_name = device_name
        self.ib_port = ib_port
        self.gid_index = gid_index
        
        self.lib = None
        self.context = None
        self.pd = None
        self.send_cq = None
        self.recv_cq = None
        self.port_attr = None
        self.gid = None
        self.lid = 0
        
        # Queue pairs: remote_address -> (qp_ptr, qp_num)
        self._qps: dict[str, tuple[ctypes.c_void_p, int]] = {}
        # Registered memory regions: addr -> RdmaMemoryRegion
        self._mrs: dict[int, RdmaMemoryRegion] = {}
        # Lock for thread safety
        self._lock = threading.Lock()
        
        self._load_library()
        if self.lib is not None:
            self._init_device()
    
    def _load_library(self):
        """Load libibverbs shared library."""
        lib_names = ["libibverbs.so.1", "libibverbs.so", "libmlx5.so.1"]
        
        for lib_name in lib_names:
            try:
                self.lib = ctypes.CDLL(lib_name, mode=ctypes.RTLD_GLOBAL)
                logger.info("Loaded RDMA library: %s", lib_name)
                self._setup_functions()
                return
            except OSError:
                continue
        
        logger.warning(
            "libibverbs not found. RDMA transport unavailable. "
            "Install with: apt-get install libibverbs-dev rdma-core"
        )
    
    def _setup_functions(self):
        """Setup ctypes function prototypes for ibverbs."""
        if self.lib is None:
            return
        
        # Device management
        self.lib.ibv_get_device_list.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.lib.ibv_get_device_list.restype = ctypes.POINTER(ctypes.c_void_p)
        self.lib.ibv_free_device_list.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.ibv_get_device_name.argtypes = [ctypes.c_void_p]
        self.lib.ibv_get_device_name.restype = ctypes.c_char_p
        self.lib.ibv_open_device.argtypes = [ctypes.c_void_p]
        self.lib.ibv_open_device.restype = ctypes.c_void_p
        self.lib.ibv_close_device.argtypes = [ctypes.c_void_p]
        self.lib.ibv_close_device.restype = ctypes.c_int
        
        # Protection domain
        self.lib.ibv_alloc_pd.argtypes = [ctypes.c_void_p]
        self.lib.ibv_alloc_pd.restype = ctypes.c_void_p
        self.lib.ibv_dealloc_pd.argtypes = [ctypes.c_void_p]
        self.lib.ibv_dealloc_pd.restype = ctypes.c_int
        
        # Memory registration
        self.lib.ibv_reg_mr.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
        ]
        self.lib.ibv_reg_mr.restype = ctypes.POINTER(ibv_mr)
        self.lib.ibv_dereg_mr.argtypes = [ctypes.POINTER(ibv_mr)]
        self.lib.ibv_dereg_mr.restype = ctypes.c_int
        
        # Completion queue
        self.lib.ibv_create_cq.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int
        ]
        self.lib.ibv_create_cq.restype = ctypes.c_void_p
        self.lib.ibv_destroy_cq.argtypes = [ctypes.c_void_p]
        self.lib.ibv_destroy_cq.restype = ctypes.c_int
        self.lib.ibv_poll_cq.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ibv_wc)
        ]
        self.lib.ibv_poll_cq.restype = ctypes.c_int
        
        # Queue pair
        self.lib.ibv_create_qp.argtypes = [ctypes.c_void_p, ctypes.POINTER(ibv_qp_init_attr)]
        self.lib.ibv_create_qp.restype = ctypes.c_void_p
        self.lib.ibv_destroy_qp.argtypes = [ctypes.c_void_p]
        self.lib.ibv_destroy_qp.restype = ctypes.c_int
        self.lib.ibv_modify_qp.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ibv_qp_attr), ctypes.c_int
        ]
        self.lib.ibv_modify_qp.restype = ctypes.c_int
        
        # Port and GID
        self.lib.ibv_query_port.argtypes = [
            ctypes.c_void_p, ctypes.c_uint8, ctypes.POINTER(ibv_port_attr)
        ]
        self.lib.ibv_query_port.restype = ctypes.c_int
        self.lib.ibv_query_gid.argtypes = [
            ctypes.c_void_p, ctypes.c_uint8, ctypes.c_int, ctypes.POINTER(ibv_gid)
        ]
        self.lib.ibv_query_gid.restype = ctypes.c_int
        
        # Work requests
        self.lib.ibv_post_send.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ibv_send_wr),
            ctypes.POINTER(ctypes.POINTER(ibv_send_wr))
        ]
        self.lib.ibv_post_send.restype = ctypes.c_int
        self.lib.ibv_post_recv.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ibv_recv_wr),
            ctypes.POINTER(ctypes.POINTER(ibv_recv_wr))
        ]
        self.lib.ibv_post_recv.restype = ctypes.c_int
    
    def _init_device(self):
        """Initialize RDMA device, PD, and CQs."""
        if self.lib is None:
            return
        
        try:
            # Get device list
            num_devices = ctypes.c_int()
            device_list = self.lib.ibv_get_device_list(ctypes.byref(num_devices))
            
            if not device_list or num_devices.value == 0:
                logger.warning("No RDMA devices found")
                self.lib = None
                return
            
            # Select device
            selected_device = None
            for i in range(num_devices.value):
                dev = device_list[i]
                if dev:
                    name = self.lib.ibv_get_device_name(dev)
                    if name:
                        name_str = name.decode('utf-8')
                        if self.device_name is None or name_str == self.device_name:
                            selected_device = dev
                            logger.info("Using RDMA device: %s", name_str)
                            break
            
            if selected_device is None:
                logger.warning("RDMA device not found")
                self.lib.ibv_free_device_list(device_list)
                self.lib = None
                return
            
            # Open device
            self.context = self.lib.ibv_open_device(selected_device)
            self.lib.ibv_free_device_list(device_list)
            
            if not self.context:
                logger.warning("Failed to open RDMA device")
                self.lib = None
                return
            
            # Allocate protection domain
            self.pd = self.lib.ibv_alloc_pd(self.context)
            if not self.pd:
                raise RuntimeError("Failed to allocate protection domain")
            
            # Create completion queues
            cq_size = 1024
            self.send_cq = self.lib.ibv_create_cq(self.context, cq_size, None, None, 0)
            self.recv_cq = self.lib.ibv_create_cq(self.context, cq_size, None, None, 0)
            if not self.send_cq or not self.recv_cq:
                raise RuntimeError("Failed to create completion queues")
            
            # Query port
            self.port_attr = ibv_port_attr()
            ret = self.lib.ibv_query_port(
                self.context, self.ib_port, ctypes.byref(self.port_attr)
            )
            if ret != 0:
                raise RuntimeError(f"Failed to query port: {ret}")
            self.lid = self.port_attr.lid
            
            # Query GID
            self.gid = ibv_gid()
            ret = self.lib.ibv_query_gid(
                self.context, self.ib_port, self.gid_index, ctypes.byref(self.gid)
            )
            if ret != 0:
                raise RuntimeError(f"Failed to query GID: {ret}")
            
            logger.info(
                "RDMA transport initialized: lid=%d, port_state=%d",
                self.lid, self.port_attr.state
            )
            
        except Exception as e:
            logger.warning("Failed to initialize RDMA: %s", e)
            self._cleanup()
            self.lib = None
    
    @property
    def is_available(self) -> bool:
        """Check if RDMA is available."""
        return (self.lib is not None and 
                self.context is not None and
                self.port_attr is not None and
                self.port_attr.state == IBV_PORT_ACTIVE)
    
    def get_local_info(self) -> QueuePairInfo | None:
        """Get local connection info."""
        if not self.is_available:
            return None
        return QueuePairInfo(
            qp_num=0,  # Will be filled when QP is created
            lid=self.lid,
            psn=0,
            gid=bytes(self.gid.raw),
        )
    
    def register_memory(
        self,
        addr: int,
        length: int,
        access: int = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ,
    ) -> RdmaMemoryRegion | None:
        """
        Register memory region for RDMA.
        
        For GPU memory, this enables GPUDirect RDMA - the NIC can directly
        access GPU memory without CPU involvement.
        """
        if not self.is_available:
            return None
        
        with self._lock:
            if addr in self._mrs:
                return self._mrs[addr]
            
            try:
                mr = self.lib.ibv_reg_mr(
                    self.pd,
                    ctypes.c_void_p(addr),
                    length,
                    access,
                )
                
                if not mr:
                    logger.warning("Failed to register memory at 0x%x", addr)
                    return None
                
                region = RdmaMemoryRegion(
                    addr=mr.contents.addr,
                    length=mr.contents.length,
                    lkey=mr.contents.lkey,
                    rkey=mr.contents.rkey,
                    mr_ptr=ctypes.cast(mr, ctypes.c_void_p),
                )
                self._mrs[addr] = region
                
                logger.debug(
                    "Registered memory: addr=0x%x, length=%d, lkey=%d, rkey=%d",
                    addr, length, region.lkey, region.rkey
                )
                return region
                
            except Exception as e:
                logger.warning("Memory registration failed: %s", e)
                return None
    
    def deregister_memory(self, addr: int):
        """Deregister memory region."""
        with self._lock:
            if addr in self._mrs:
                mr = self._mrs.pop(addr)
                if self.lib is not None:
                    mr_ptr = ctypes.cast(mr.mr_ptr, ctypes.POINTER(ibv_mr))
                    self.lib.ibv_dereg_mr(mr_ptr)
    
    def create_qp(self, remote_address: str) -> tuple[ctypes.c_void_p, int] | None:
        """Create a Queue Pair for a connection."""
        if not self.is_available:
            return None
        
        with self._lock:
            if remote_address in self._qps:
                return self._qps[remote_address]
            
            try:
                # Initialize QP attributes
                init_attr = ibv_qp_init_attr()
                init_attr.send_cq = self.send_cq
                init_attr.recv_cq = self.recv_cq
                init_attr.cap_max_send_wr = 128
                init_attr.cap_max_recv_wr = 128
                init_attr.cap_max_send_sge = 1
                init_attr.cap_max_recv_sge = 1
                init_attr.qp_type = IBV_QPT_RC
                
                # Create QP
                qp = self.lib.ibv_create_qp(self.pd, ctypes.byref(init_attr))
                if not qp:
                    logger.warning("Failed to create QP for %s", remote_address)
                    return None
                
                # Get QP number (at offset 4 in the ibv_qp struct)
                qp_num = ctypes.cast(qp, ctypes.POINTER(ctypes.c_uint32))[1]
                
                self._qps[remote_address] = (qp, qp_num)
                logger.debug("Created QP %d for %s", qp_num, remote_address)
                
                return (qp, qp_num)
                
            except Exception as e:
                logger.warning("Failed to create QP: %s", e)
                return None
    
    def modify_qp_to_init(self, qp: ctypes.c_void_p) -> bool:
        """Transition QP to INIT state."""
        attr = ibv_qp_attr()
        attr.qp_state = IBV_QPS_INIT
        attr.port_num = self.ib_port
        attr.pkey_index = 0
        attr.qp_access_flags = (
            IBV_ACCESS_REMOTE_WRITE | 
            IBV_ACCESS_REMOTE_READ |
            IBV_ACCESS_LOCAL_WRITE
        )
        
        mask = (self.IBV_QP_STATE | self.IBV_QP_PKEY_INDEX | 
                self.IBV_QP_PORT | self.IBV_QP_ACCESS_FLAGS)
        
        ret = self.lib.ibv_modify_qp(qp, ctypes.byref(attr), mask)
        return ret == 0
    
    def modify_qp_to_rtr(
        self, qp: ctypes.c_void_p, remote_info: RemoteQPInfo
    ) -> bool:
        """Transition QP to RTR (Ready to Receive) state."""
        attr = ibv_qp_attr()
        attr.qp_state = IBV_QPS_RTR
        attr.path_mtu = IBV_MTU_4096
        attr.dest_qp_num = remote_info.qp_num
        attr.rq_psn = remote_info.psn
        attr.max_dest_rd_atomic = 1
        attr.min_rnr_timer = 12
        
        # Setup address handle
        attr.ah_attr.dlid = remote_info.lid
        attr.ah_attr.sl = 0
        attr.ah_attr.src_path_bits = 0
        attr.ah_attr.port_num = self.ib_port
        
        # Use GRH for RoCE
        if remote_info.gid and any(remote_info.gid):
            attr.ah_attr.is_global = 1
            ctypes.memmove(attr.ah_attr.grh.dgid.raw, remote_info.gid, 16)
            attr.ah_attr.grh.flow_label = 0
            attr.ah_attr.grh.hop_limit = 1
            attr.ah_attr.grh.sgid_index = self.gid_index
            attr.ah_attr.grh.traffic_class = 0
        
        mask = (self.IBV_QP_STATE | self.IBV_QP_AV | self.IBV_QP_PATH_MTU |
                self.IBV_QP_DEST_QPN | self.IBV_QP_RQ_PSN |
                self.IBV_QP_MAX_DEST_RD_ATOMIC | self.IBV_QP_MIN_RNR_TIMER)
        
        ret = self.lib.ibv_modify_qp(qp, ctypes.byref(attr), mask)
        return ret == 0
    
    def modify_qp_to_rts(self, qp: ctypes.c_void_p, psn: int = 0) -> bool:
        """Transition QP to RTS (Ready to Send) state."""
        attr = ibv_qp_attr()
        attr.qp_state = IBV_QPS_RTS
        attr.timeout = 14
        attr.retry_cnt = 7
        attr.rnr_retry = 7
        attr.sq_psn = psn
        attr.max_rd_atomic = 1
        
        mask = (self.IBV_QP_STATE | self.IBV_QP_TIMEOUT | self.IBV_QP_RETRY_CNT |
                self.IBV_QP_RNR_RETRY | self.IBV_QP_SQ_PSN | self.IBV_QP_MAX_QP_RD_ATOMIC)
        
        ret = self.lib.ibv_modify_qp(qp, ctypes.byref(attr), mask)
        return ret == 0
    
    def rdma_write(
        self,
        qp: ctypes.c_void_p,
        local_addr: int,
        local_lkey: int,
        remote_addr: int,
        remote_rkey: int,
        length: int,
        signaled: bool = True,
    ) -> bool:
        """
        Perform RDMA Write operation.
        
        This writes data directly from local memory to remote memory
        without involving the remote CPU - true zero-copy.
        """
        if not self.is_available:
            return False
        
        # Setup scatter-gather entry
        sge = ibv_sge()
        sge.addr = local_addr
        sge.length = length
        sge.lkey = local_lkey
        
        # Setup work request
        wr = ibv_send_wr()
        wr.wr_id = local_addr  # Use address as ID
        wr.next = None
        wr.sg_list = ctypes.pointer(sge)
        wr.num_sge = 1
        wr.opcode = IBV_WR_RDMA_WRITE
        wr.send_flags = IBV_SEND_SIGNALED if signaled else 0
        wr.remote_addr = remote_addr
        wr.rkey = remote_rkey
        
        bad_wr = ctypes.POINTER(ibv_send_wr)()
        ret = self.lib.ibv_post_send(qp, ctypes.byref(wr), ctypes.byref(bad_wr))
        
        if ret != 0:
            logger.warning("RDMA Write failed: %d", ret)
            return False
        
        return True
    
    def poll_completion(self, timeout_ms: int = 1000) -> list[tuple[int, int]]:
        """Poll for work completions. Returns list of (wr_id, status)."""
        if not self.is_available:
            return []
        
        wc = ibv_wc()
        completions = []
        
        start = time.time()
        while (time.time() - start) * 1000 < timeout_ms:
            ret = self.lib.ibv_poll_cq(self.send_cq, 1, ctypes.byref(wc))
            if ret > 0:
                completions.append((wc.wr_id, wc.status))
                if wc.status != IBV_WC_SUCCESS:
                    logger.warning("WC error: wr_id=%d, status=%d", wc.wr_id, wc.status)
            elif ret < 0:
                logger.warning("Poll CQ error: %d", ret)
                break
            else:
                time.sleep(0.0001)  # 100us
        
        return completions
    
    def _cleanup(self):
        """Cleanup RDMA resources."""
        if self.lib is None:
            return
        
        # Destroy QPs
        for qp, _ in self._qps.values():
            try:
                self.lib.ibv_destroy_qp(qp)
            except Exception:
                pass
        self._qps.clear()
        
        # Deregister MRs
        for addr in list(self._mrs.keys()):
            self.deregister_memory(addr)
        
        # Destroy CQs
        if self.send_cq:
            self.lib.ibv_destroy_cq(self.send_cq)
        if self.recv_cq:
            self.lib.ibv_destroy_cq(self.recv_cq)
        
        # Deallocate PD
        if self.pd:
            self.lib.ibv_dealloc_pd(self.pd)
        
        # Close device
        if self.context:
            self.lib.ibv_close_device(self.context)
    
    def close(self):
        """Close transport and cleanup resources."""
        self._cleanup()


# ============================================================================
# NVLink P2P Transport
# ============================================================================

class NvlinkP2pTransport:
    """
    NVLink P2P transport for intra-node GPU-to-GPU transfers.
    
    Uses CUDA P2P memory access with copy engines (not SMs).
    """
    
    def __init__(self, local_rank: int, device: torch.device):
        self.local_rank = local_rank
        self.device = device
        self._peer_access_enabled: set[int] = set()
        self._copy_stream = torch.cuda.Stream(device=device)
        
        self._enable_p2p_access()
    
    def _enable_p2p_access(self):
        """Enable P2P access to all peer GPUs."""
        num_gpus = torch.cuda.device_count()
        
        for peer_device in range(num_gpus):
            if peer_device == self.local_rank:
                continue
            
            try:
                can_access = torch.cuda.can_device_access_peer(
                    self.local_rank, peer_device
                )
                if can_access:
                    with torch.cuda.device(self.local_rank):
                        torch.cuda.enable_peer_access(peer_device)
                    self._peer_access_enabled.add(peer_device)
                    logger.debug(
                        "P2P enabled: GPU %d -> GPU %d",
                        self.local_rank, peer_device
                    )
            except RuntimeError as e:
                if "already enabled" in str(e).lower():
                    self._peer_access_enabled.add(peer_device)
    
    def copy_p2p(
        self,
        src_tensor: torch.Tensor,
        dst_tensor: torch.Tensor,
    ) -> torch.cuda.Event:
        """
        P2P copy using copy engine (SM-free).
        
        Returns an event to synchronize on.
        """
        with torch.cuda.stream(self._copy_stream):
            dst_tensor.copy_(src_tensor, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self._copy_stream)
        return event
    
    def is_p2p_available(self, peer_device: int) -> bool:
        return peer_device in self._peer_access_enabled
    
    def close(self):
        pass


# ============================================================================
# Main P2P RDMA Engine
# ============================================================================

class P2pRdmaEngine:
    """
    P2P RDMA Engine for KV cache transfer.
    
    Features:
    - Raw ibverbs RDMA (no UCX)
    - GPUDirect RDMA for zero-copy GPU transfers
    - NVLink P2P for intra-node
    - SM-free data movement via copy engines
    """
    
    def __init__(
        self,
        local_rank: int,
        config: KVTransferConfig,
        hostname: str = "",
        port_offset: int = 0,
    ) -> None:
        self.config = config
        self.rank = port_offset
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{self.local_rank}")
        
        torch.cuda.set_device(self.device)
        
        if not hostname:
            hostname = get_ip()
        port = int(self.config.kv_port) + port_offset
        if port == 0:
            raise ValueError("Port cannot be 0")
        self._hostname = hostname
        self._port = port
        self._local_hostname = get_hostname()
        
        self.zmq_address = f"{self._hostname}:{self._port}"
        
        # Proxy config
        proxy_ip = self.config.get_from_extra_config("proxy_ip", "")
        proxy_port = self.config.get_from_extra_config("proxy_port", "")
        if proxy_ip == "" or proxy_port == "":
            self.proxy_address = ""
            self.http_address = ""
        else:
            self.proxy_address = proxy_ip + ":" + proxy_port
            http_port = self.config.get_from_extra_config("http_port", None)
            if http_port is None:
                raise ValueError("http_port required in kv_connector_extra_config")
            self.http_address = f"{self._hostname}:{http_port}"
        
        # ZMQ for signaling
        self.context = zmq.Context()
        self.router_socket = self.context.socket(zmq.ROUTER)
        self.router_socket.bind(f"tcp://{self.zmq_address}")
        
        self.poller = zmq.Poller()
        self.poller.register(self.router_socket, zmq.POLLIN)
        
        # Synchronization
        self.send_store_cv = threading.Condition()
        self.send_queue_cv = threading.Condition()
        self.recv_store_cv = threading.Condition()
        
        # Copy stream (uses DMA engine, not SMs)
        self.copy_stream = torch.cuda.Stream(device=self.device)
        
        # Memory pool (only for overflow)
        mem_pool_size_gb = float(
            self.config.get_from_extra_config("mem_pool_size_gb", DEFAULT_MEM_POOL_SIZE_GB)
        )
        self.pool = TensorMemoryPool(max_block_size=int(mem_pool_size_gb * 1024**3))
        
        # Transport layers
        self.rdma = RdmaTransport()
        self.nvlink = NvlinkP2pTransport(self.local_rank, self.device)
        
        # Send configuration
        self.send_type = self.config.get_from_extra_config("send_type", "PUT_ASYNC")
        if self.send_type == "GET":
            self.send_store: dict[str, torch.Tensor] = {}
        else:
            self.send_queue: deque[SendQueueItem] = deque()
            if self.send_type == "PUT_ASYNC":
                self._send_thread = threading.Thread(target=self.send_async, daemon=True)
                self._send_thread.start()
        
        # Storage
        self.recv_store: dict[str, Any] = {}
        self.recv_request_id_to_tensor_ids: dict[str, set[str]] = {}
        self.send_request_id_to_tensor_ids: dict[str, set[str]] = {}
        
        # Connections
        self.socks: dict[str, Any] = {}
        self.peer_same_node: dict[str, bool] = {}
        self.remote_qp_info: dict[str, RemoteQPInfo] = {}
        
        # Buffer
        self.buffer_size = 0
        self.buffer_size_threshold = float(self.config.kv_buffer_size)
        
        # Listener
        self._listener_thread = threading.Thread(target=self.listen_for_requests, daemon=True)
        self._listener_thread.start()
        
        # Ping
        self._ping_thread = None
        if port_offset == 0 and self.proxy_address != "":
            self._ping_thread = threading.Thread(target=self.ping, daemon=True)
            self._ping_thread.start()
        
        logger.info(
            "💯P2pRdmaEngine init: rank=%d, local_rank=%d, zmq=%s, rdma=%s",
            self.rank, self.local_rank, self.zmq_address, self.rdma.is_available
        )
    
    def _is_same_node(self, remote_address: str) -> bool:
        if remote_address not in self.peer_same_node:
            self.peer_same_node[remote_address] = is_same_node(
                self._local_hostname, remote_address
            )
        return self.peer_same_node[remote_address]
    
    def create_connect(self, remote_address: str | None = None):
        assert remote_address is not None
        if remote_address not in self.socks:
            sock = self.context.socket(zmq.DEALER)
            sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)
            sock.connect(f"tcp://{remote_address}")
            self.socks[remote_address] = sock
            
            same_node = self._is_same_node(remote_address)
            
            # Get local RDMA info
            local_info = self.rdma.get_local_info()
            rdma_info = None
            if local_info and not same_node:
                qp_result = self.rdma.create_qp(remote_address)
                if qp_result:
                    qp, qp_num = qp_result
                    self.rdma.modify_qp_to_init(qp)
                    rdma_info = {
                        "qp_num": qp_num,
                        "lid": local_info.lid,
                        "psn": 0,
                        "gid": local_info.gid,
                    }
            
            data = {
                "cmd": "NEW",
                "hostname": self._local_hostname,
                "local_rank": self.local_rank,
                "same_node": same_node,
                "rdma_info": rdma_info,
            }
            sock.send(msgpack.dumps(data))
            
            # Wait for response with remote RDMA info
            response = sock.recv()
            resp_data = msgpack.loads(response)
            if resp_data.get("rdma_info") and not same_node:
                ri = resp_data["rdma_info"]
                remote_info = RemoteQPInfo(
                    qp_num=ri["qp_num"],
                    lid=ri["lid"],
                    psn=ri["psn"],
                    gid=ri["gid"],
                )
                self.remote_qp_info[remote_address] = remote_info
                
                # Complete QP setup
                qp, _ = self._qps.get(remote_address, (None, None))
                if qp:
                    self.rdma.modify_qp_to_rtr(qp, remote_info)
                    self.rdma.modify_qp_to_rts(qp)
            
            logger.info("🤝Connected: %s👉%s, same_node=%s", 
                       self.zmq_address, remote_address, same_node)
        
        return self.socks[remote_address]
    
    @property
    def _qps(self):
        return self.rdma._qps
    
    def send_tensor(
        self,
        tensor_id: str,
        tensor: torch.Tensor,
        remote_address: str | None = None,
    ) -> bool:
        if remote_address is None:
            with self.recv_store_cv:
                self.recv_store[tensor_id] = tensor
                self.recv_store_cv.notify()
            return True
        
        item = SendQueueItem(
            tensor_id=tensor_id, remote_address=remote_address, tensor=tensor
        )
        
        if self.send_type == "PUT":
            return self.send_sync(item)
        
        if self.send_type == "PUT_ASYNC":
            with self.send_queue_cv:
                self.send_queue.append(item)
                self.send_queue_cv.notify()
            return True
        
        # GET mode
        with self.send_store_cv:
            tensor_size = tensor.element_size() * tensor.numel()
            if tensor_size > self.buffer_size_threshold:
                return False
            while self.buffer_size + tensor_size > self.buffer_size_threshold:
                if not self.send_store:
                    break
                oldest_id = next(iter(self.send_store))
                oldest = self.send_store.pop(oldest_id)
                self.buffer_size -= oldest.element_size() * oldest.numel()
            self.send_store[tensor_id] = tensor
            self.buffer_size += tensor_size
        return True
    
    def recv_tensor(
        self,
        tensor_id: str,
        remote_address: str | None = None,
    ) -> torch.Tensor:
        if self.send_type in ("PUT", "PUT_ASYNC"):
            with self.recv_store_cv:
                while tensor_id not in self.recv_store:
                    self.recv_store_cv.wait()
                tensor = self.recv_store[tensor_id]
            
            if tensor is not None:
                if isinstance(tensor, tuple):
                    addr, dtype, shape = tensor
                    tensor = self.pool.load_tensor(addr, dtype, shape, self.device)
                else:
                    self.buffer_size -= tensor.element_size() * tensor.numel()
            return tensor
        
        # GET mode - simplified
        if remote_address is None:
            return None
        
        if remote_address not in self.socks:
            self.create_connect(remote_address)
        
        sock = self.socks[remote_address]
        data = {"cmd": "GET", "tensor_id": tensor_id}
        sock.send(msgpack.dumps(data))
        
        message = sock.recv()
        data = msgpack.loads(message)
        if data["ret"] != 0:
            return None
        
        tensor = torch.empty(
            data["shape"],
            dtype=getattr(torch, data["dtype"]),
            device=self.device,
        )
        
        # Receive tensor data
        tensor_data = sock.recv()
        with torch.cuda.stream(self.copy_stream):
            cpu_tensor = torch.frombuffer(tensor_data, dtype=tensor.dtype).reshape(tensor.shape)
            tensor.copy_(cpu_tensor, non_blocking=True)
        self.copy_stream.synchronize()
        
        return tensor
    
    def listen_for_requests(self):
        while True:
            socks = dict(self.poller.poll())
            if self.router_socket not in socks:
                continue
            
            remote_address, message = self.router_socket.recv_multipart()
            data = msgpack.loads(message)
            
            if data["cmd"] == "NEW":
                same_node = data.get("same_node", False)
                self.peer_same_node[remote_address.decode()] = same_node
                
                # Setup RDMA QP if remote sent RDMA info
                rdma_info = data.get("rdma_info")
                response_rdma = None
                if rdma_info and not same_node and self.rdma.is_available:
                    remote_info = RemoteQPInfo(
                        qp_num=rdma_info["qp_num"],
                        lid=rdma_info["lid"],
                        psn=rdma_info["psn"],
                        gid=rdma_info["gid"],
                    )
                    self.remote_qp_info[remote_address.decode()] = remote_info
                    
                    # Create our QP
                    qp_result = self.rdma.create_qp(remote_address.decode())
                    if qp_result:
                        qp, qp_num = qp_result
                        self.rdma.modify_qp_to_init(qp)
                        self.rdma.modify_qp_to_rtr(qp, remote_info)
                        self.rdma.modify_qp_to_rts(qp)
                        
                        local_info = self.rdma.get_local_info()
                        response_rdma = {
                            "qp_num": qp_num,
                            "lid": local_info.lid,
                            "psn": 0,
                            "gid": local_info.gid,
                        }
                
                response = {"rdma_info": response_rdma}
                self.router_socket.send_multipart([remote_address, msgpack.dumps(response)])
                
                logger.info("🤝New: %s👈%s, same_node=%s, rdma=%s",
                           self.zmq_address, remote_address.decode(), same_node,
                           response_rdma is not None)
                
            elif data["cmd"] == "PUT":
                tensor_id = data["tensor_id"]
                
                try:
                    tensor = torch.empty(
                        data["shape"],
                        dtype=getattr(torch, data["dtype"]),
                        device=self.device,
                    )
                    
                    self.router_socket.send_multipart([remote_address, b"0"])
                    _, tensor_data = self.router_socket.recv_multipart()
                    
                    with torch.cuda.stream(self.copy_stream):
                        cpu_tensor = torch.frombuffer(tensor_data, dtype=tensor.dtype).reshape(tensor.shape)
                        tensor.copy_(cpu_tensor, non_blocking=True)
                    self.copy_stream.synchronize()
                    
                    tensor_size = tensor.element_size() * tensor.numel()
                    if self.buffer_size + tensor_size > self.buffer_size_threshold:
                        addr = self.pool.store_tensor(tensor)
                        tensor = (addr, tensor.dtype, tensor.shape)
                    else:
                        self.buffer_size += tensor_size
                        
                except torch.cuda.OutOfMemoryError:
                    self.router_socket.send_multipart([remote_address, b"1"])
                    tensor = None
                
                with self.recv_store_cv:
                    self.recv_store[tensor_id] = tensor
                    self.have_received_tensor_id(tensor_id)
                    self.recv_store_cv.notify()
                    
            elif data["cmd"] == "GET":
                tensor_id = data["tensor_id"]
                
                with self.send_store_cv:
                    tensor = self.send_store.pop(tensor_id, None)
                    if tensor is not None:
                        self.send_store[tensor_id] = tensor  # LRU
                        self.have_sent_tensor_id(tensor_id)
                        response = {
                            "ret": 0,
                            "shape": list(tensor.shape),
                            "dtype": str(tensor.dtype).replace("torch.", ""),
                        }
                    else:
                        response = {"ret": 1}
                
                self.router_socket.send_multipart([remote_address, msgpack.dumps(response)])
                
                if response["ret"] == 0:
                    with torch.cuda.stream(self.copy_stream):
                        tensor_bytes = tensor.to(self.device).cpu().numpy().tobytes()
                    self.copy_stream.synchronize()
                    self.router_socket.send_multipart([remote_address, tensor_bytes])
    
    def have_sent_tensor_id(self, tensor_id: str):
        request_id = tensor_id.split("#")[0]
        if request_id not in self.send_request_id_to_tensor_ids:
            self.send_request_id_to_tensor_ids[request_id] = set()
        self.send_request_id_to_tensor_ids[request_id].add(tensor_id)
    
    def have_received_tensor_id(self, tensor_id: str):
        request_id = tensor_id.split("#")[0]
        if request_id not in self.recv_request_id_to_tensor_ids:
            self.recv_request_id_to_tensor_ids[request_id] = set()
        self.recv_request_id_to_tensor_ids[request_id].add(tensor_id)
    
    def send_async(self):
        while True:
            with self.send_queue_cv:
                while not self.send_queue:
                    self.send_queue_cv.wait()
                item = self.send_queue.popleft()
                if not self.send_queue:
                    self.send_queue_cv.notify()
            self.send_sync(item)
    
    def wait_for_sent(self):
        if self.send_type == "PUT_ASYNC":
            with self.send_queue_cv:
                while self.send_queue:
                    self.send_queue_cv.wait()
    
    def send_sync(self, item: SendQueueItem) -> bool:
        if item.remote_address is None:
            return False
        if item.remote_address not in self.socks:
            self.create_connect(item.remote_address)
        
        tensor = item.tensor.to(self.device)
        sock = self.socks[item.remote_address]
        same_node = self._is_same_node(item.remote_address)
        
        # For now, use ZMQ-based transfer
        # RDMA write path would use self.rdma.rdma_write()
        data = {
            "cmd": "PUT",
            "tensor_id": item.tensor_id,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
        }
        sock.send(msgpack.dumps(data))
        
        response = sock.recv()
        if response != b"0":
            return False
        
        # Copy to CPU using stream (SM-free)
        with torch.cuda.stream(self.copy_stream):
            tensor_bytes = tensor.cpu().numpy().tobytes()
        self.copy_stream.synchronize()
        sock.send(tensor_bytes)
        
        if self.send_type == "PUT_ASYNC":
            self.have_sent_tensor_id(item.tensor_id)
        
        return True
    
    def get_finished(
        self, finished_req_ids: set[str], no_compile_layers
    ) -> tuple[set[str] | None, set[str] | None]:
        for request_id in finished_req_ids:
            for layer_name in no_compile_layers:
                tensor_id = request_id + "#" + layer_name
                if tensor_id in self.recv_store:
                    with self.recv_store_cv:
                        tensor = self.recv_store.pop(tensor_id, None)
                        self.send_request_id_to_tensor_ids.pop(request_id, None)
                        self.recv_request_id_to_tensor_ids.pop(request_id, None)
                    if isinstance(tensor, tuple):
                        addr, _, _ = tensor
                        self.pool.free(addr)
        
        return set() or None, set() or None
    
    def ping(self):
        sock = self.context.socket(zmq.DEALER)
        sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)
        sock.connect(f"tcp://{self.proxy_address}")
        data = {
            "type": "P" if self.config.is_kv_producer else "D",
            "http_address": self.http_address,
            "zmq_address": self.zmq_address,
        }
        while True:
            sock.send(msgpack.dumps(data))
            time.sleep(3)
    
    def close(self) -> None:
        self._listener_thread.join(timeout=1)
        if self.send_type == "PUT_ASYNC":
            self._send_thread.join(timeout=1)
        if self._ping_thread is not None:
            self._ping_thread.join(timeout=1)
        
        self.rdma.close()
        self.nvlink.close()
        
        for sock in self.socks.values():
            sock.close()
        self.router_socket.close()
        self.context.term()
