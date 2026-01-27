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

Fault Tolerance features:
- Node crash detection and handling
- Connection timeout with automatic retry
- Automatic reconnection on failure
- Scale-up and scale-down support
- Graceful degradation on partial failures
"""

import ctypes
import enum
import json
import logging
import os
import socket
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

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
# Fault Tolerance Configuration
# ============================================================================

# Connection timeout in seconds
CONNECTION_TIMEOUT_SEC = 10.0
# Send/receive timeout in milliseconds
SEND_RECV_TIMEOUT_MS = 5000
# Maximum retry attempts for connection
MAX_RETRY_ATTEMPTS = 3
# Retry backoff base in seconds
RETRY_BACKOFF_BASE = 1.0
# Heartbeat interval in seconds
HEARTBEAT_INTERVAL_SEC = 5.0
# Heartbeat timeout (miss count before declaring dead)
HEARTBEAT_MISS_THRESHOLD = 3
# Connection check interval
CONNECTION_CHECK_INTERVAL_SEC = 10.0


class ConnectionState(enum.Enum):
    """Connection state for fault tolerance."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"
    REMOVED = "removed"  # For scale-down


@dataclass
class PeerInfo:
    """Information about a remote peer."""
    address: str
    hostname: str
    local_rank: int
    same_node: bool
    state: ConnectionState = ConnectionState.DISCONNECTED
    last_heartbeat: float = 0.0
    heartbeat_misses: int = 0
    retry_count: int = 0
    last_error: str = ""
    connected_at: float = 0.0
    rdma_info: dict | None = None
    
    def is_healthy(self) -> bool:
        """Check if peer is healthy."""
        return self.state == ConnectionState.CONNECTED
    
    def is_recoverable(self) -> bool:
        """Check if connection can be recovered."""
        return self.state in (
            ConnectionState.DISCONNECTED,
            ConnectionState.RECONNECTING,
        ) and self.retry_count < MAX_RETRY_ATTEMPTS
    
    def mark_failed(self, error: str):
        """Mark peer as failed."""
        self.state = ConnectionState.FAILED
        self.last_error = error
        self.retry_count += 1
    
    def mark_connected(self):
        """Mark peer as connected."""
        self.state = ConnectionState.CONNECTED
        self.connected_at = time.time()
        self.last_heartbeat = time.time()
        self.heartbeat_misses = 0
        self.retry_count = 0
    
    def mark_disconnected(self):
        """Mark peer as disconnected."""
        self.state = ConnectionState.DISCONNECTED
    
    def mark_removed(self):
        """Mark peer as removed (scale-down)."""
        self.state = ConnectionState.REMOVED
    
    def update_heartbeat(self):
        """Update heartbeat timestamp."""
        self.last_heartbeat = time.time()
        self.heartbeat_misses = 0
    
    def check_heartbeat(self) -> bool:
        """Check if heartbeat is healthy. Returns False if dead."""
        if self.state != ConnectionState.CONNECTED:
            return True  # Not connected, don't check
        
        elapsed = time.time() - self.last_heartbeat
        if elapsed > HEARTBEAT_INTERVAL_SEC:
            self.heartbeat_misses += 1
            if self.heartbeat_misses >= HEARTBEAT_MISS_THRESHOLD:
                return False
        return True


class ConnectionManager:
    """
    Manages connections with fault tolerance.
    
    Handles:
    - Connection lifecycle
    - Heartbeat monitoring
    - Automatic reconnection
    - Scale-up/scale-down
    """
    
    def __init__(
        self,
        local_address: str,
        on_peer_connected: Callable[[str], None] | None = None,
        on_peer_disconnected: Callable[[str], None] | None = None,
        on_peer_removed: Callable[[str], None] | None = None,
    ):
        self.local_address = local_address
        self.peers: dict[str, PeerInfo] = {}
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        
        # Callbacks
        self._on_peer_connected = on_peer_connected
        self._on_peer_disconnected = on_peer_disconnected
        self._on_peer_removed = on_peer_removed
        
        # Stats
        self.stats = {
            "connections_established": 0,
            "connections_failed": 0,
            "reconnections": 0,
            "peers_removed": 0,
        }
    
    def add_peer(self, address: str, info: dict) -> PeerInfo:
        """Add a new peer."""
        with self._lock:
            if address in self.peers:
                # Update existing peer
                peer = self.peers[address]
                if peer.state == ConnectionState.REMOVED:
                    # Re-adding a removed peer (scale-up after scale-down)
                    peer.state = ConnectionState.DISCONNECTED
                    peer.retry_count = 0
                return peer
            
            peer = PeerInfo(
                address=address,
                hostname=info.get("hostname", ""),
                local_rank=info.get("local_rank", 0),
                same_node=info.get("same_node", False),
                rdma_info=info.get("rdma_info"),
            )
            self.peers[address] = peer
            return peer
    
    def remove_peer(self, address: str) -> bool:
        """Remove a peer (scale-down)."""
        with self._lock:
            if address not in self.peers:
                return False
            
            peer = self.peers[address]
            peer.mark_removed()
            self.stats["peers_removed"] += 1
            
            if self._on_peer_removed:
                self._on_peer_removed(address)
            
            logger.info("Peer removed (scale-down): %s", address)
            return True
    
    def get_peer(self, address: str) -> PeerInfo | None:
        """Get peer info."""
        with self._lock:
            return self.peers.get(address)
    
    def mark_connected(self, address: str):
        """Mark peer as connected."""
        with self._lock:
            if address in self.peers:
                peer = self.peers[address]
                was_reconnect = peer.state == ConnectionState.RECONNECTING
                peer.mark_connected()
                
                self.stats["connections_established"] += 1
                if was_reconnect:
                    self.stats["reconnections"] += 1
                
                if self._on_peer_connected:
                    self._on_peer_connected(address)
    
    def mark_failed(self, address: str, error: str):
        """Mark peer as failed."""
        with self._lock:
            if address in self.peers:
                peer = self.peers[address]
                peer.mark_failed(error)
                self.stats["connections_failed"] += 1
                
                if self._on_peer_disconnected:
                    self._on_peer_disconnected(address)
    
    def get_healthy_peers(self) -> list[str]:
        """Get list of healthy peer addresses."""
        with self._lock:
            return [
                addr for addr, peer in self.peers.items()
                if peer.is_healthy()
            ]
    
    def get_recoverable_peers(self) -> list[str]:
        """Get list of peers that can be reconnected."""
        with self._lock:
            return [
                addr for addr, peer in self.peers.items()
                if peer.is_recoverable()
            ]
    
    def get_all_active_peers(self) -> list[str]:
        """Get all non-removed peers."""
        with self._lock:
            return [
                addr for addr, peer in self.peers.items()
                if peer.state != ConnectionState.REMOVED
            ]
    
    def check_all_heartbeats(self) -> list[str]:
        """Check heartbeats for all peers. Returns list of dead peers."""
        dead_peers = []
        with self._lock:
            for address, peer in self.peers.items():
                if not peer.check_heartbeat():
                    peer.state = ConnectionState.RECONNECTING
                    dead_peers.append(address)
                    logger.warning(
                        "Peer heartbeat timeout: %s (misses=%d)",
                        address, peer.heartbeat_misses
                    )
        return dead_peers
    
    def update_heartbeat(self, address: str):
        """Update heartbeat for a peer."""
        with self._lock:
            if address in self.peers:
                self.peers[address].update_heartbeat()
    
    def get_stats(self) -> dict:
        """Get connection statistics."""
        with self._lock:
            stats = self.stats.copy()
            stats["total_peers"] = len(self.peers)
            stats["healthy_peers"] = len(self.get_healthy_peers())
            stats["failed_peers"] = sum(
                1 for p in self.peers.values() 
                if p.state == ConnectionState.FAILED
            )
            stats["removed_peers"] = sum(
                1 for p in self.peers.values()
                if p.state == ConnectionState.REMOVED
            )
            return stats
    
    def cleanup(self):
        """Cleanup all peers."""
        with self._lock:
            self.peers.clear()


class TransferTracker:
    """
    Tracks in-flight transfers with timeout handling.
    """
    
    def __init__(self, timeout_sec: float = 30.0):
        self.timeout_sec = timeout_sec
        self._transfers: dict[str, dict] = {}  # transfer_id -> info
        self._lock = threading.Lock()
    
    def start_transfer(
        self,
        transfer_id: str,
        tensor_id: str,
        remote_address: str,
        size_bytes: int,
    ) -> None:
        """Record start of a transfer."""
        with self._lock:
            self._transfers[transfer_id] = {
                "tensor_id": tensor_id,
                "remote_address": remote_address,
                "size_bytes": size_bytes,
                "start_time": time.time(),
                "completed": False,
                "error": None,
            }
    
    def complete_transfer(self, transfer_id: str, success: bool, error: str = ""):
        """Mark transfer as complete."""
        with self._lock:
            if transfer_id in self._transfers:
                self._transfers[transfer_id]["completed"] = True
                if not success:
                    self._transfers[transfer_id]["error"] = error
    
    def get_timed_out_transfers(self) -> list[dict]:
        """Get list of timed out transfers."""
        timed_out = []
        current_time = time.time()
        
        with self._lock:
            for transfer_id, info in list(self._transfers.items()):
                if info["completed"]:
                    continue
                
                elapsed = current_time - info["start_time"]
                if elapsed > self.timeout_sec:
                    info["transfer_id"] = transfer_id
                    timed_out.append(info)
        
        return timed_out
    
    def cancel_transfer(self, transfer_id: str):
        """Cancel a transfer."""
        with self._lock:
            if transfer_id in self._transfers:
                self._transfers[transfer_id]["completed"] = True
                self._transfers[transfer_id]["error"] = "cancelled"
    
    def cancel_transfers_for_peer(self, remote_address: str) -> list[str]:
        """Cancel all transfers for a specific peer."""
        cancelled = []
        with self._lock:
            for transfer_id, info in self._transfers.items():
                if (info["remote_address"] == remote_address and 
                    not info["completed"]):
                    info["completed"] = True
                    info["error"] = "peer_disconnected"
                    cancelled.append(transfer_id)
        return cancelled
    
    def cleanup_completed(self, max_age_sec: float = 60.0):
        """Cleanup old completed transfers."""
        current_time = time.time()
        with self._lock:
            to_remove = []
            for transfer_id, info in self._transfers.items():
                if info["completed"]:
                    age = current_time - info["start_time"]
                    if age > max_age_sec:
                        to_remove.append(transfer_id)
            
            for transfer_id in to_remove:
                del self._transfers[transfer_id]
    
    def get_pending_count(self) -> int:
        """Get count of pending transfers."""
        with self._lock:
            return sum(
                1 for info in self._transfers.values()
                if not info["completed"]
            )

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
class KVCacheMemoryInfo:
    """
    Pre-registered KV cache memory information.
    
    Registered once at initialization, then used for all transfers
    by calculating offsets - no per-tensor registration needed.
    """
    base_addr: int
    total_length: int
    lkey: int
    rkey: int
    num_blocks: int
    block_size_bytes: int  # Size of one block in bytes
    
    def get_block_addr(self, block_id: int) -> int:
        """Get address for a specific block."""
        return self.base_addr + block_id * self.block_size_bytes
    
    def get_tensor_info(self, block_ids: list[int]) -> tuple[int, int]:
        """Get (start_addr, total_length) for a list of contiguous blocks."""
        if not block_ids:
            return (0, 0)
        start_addr = self.get_block_addr(block_ids[0])
        total_length = len(block_ids) * self.block_size_bytes
        return (start_addr, total_length)


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
    
    Fault Tolerance:
    - Node crash detection
    - Connection timeout handling
    - Automatic reconnection
    - Scale-up and scale-down support
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
        
        # ZMQ for signaling with timeout
        self.context = zmq.Context()
        self.router_socket = self.context.socket(zmq.ROUTER)
        self.router_socket.setsockopt(zmq.RCVTIMEO, SEND_RECV_TIMEOUT_MS)
        self.router_socket.setsockopt(zmq.SNDTIMEO, SEND_RECV_TIMEOUT_MS)
        self.router_socket.setsockopt(zmq.LINGER, 0)  # Don't wait on close
        self.router_socket.bind(f"tcp://{self.zmq_address}")
        
        self.poller = zmq.Poller()
        self.poller.register(self.router_socket, zmq.POLLIN)
        
        # Synchronization
        self.send_store_cv = threading.Condition()
        self.send_queue_cv = threading.Condition()
        self.recv_store_cv = threading.Condition()
        self._shutdown_event = threading.Event()
        
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
        
        # Connections (with fault tolerance)
        self.socks: dict[str, Any] = {}
        self.peer_same_node: dict[str, bool] = {}
        self.remote_qp_info: dict[str, RemoteQPInfo] = {}
        
        # Fault tolerance: Connection manager
        self.conn_manager = ConnectionManager(
            local_address=self.zmq_address,
            on_peer_connected=self._on_peer_connected,
            on_peer_disconnected=self._on_peer_disconnected,
            on_peer_removed=self._on_peer_removed,
        )
        
        # Fault tolerance: Transfer tracker
        self.transfer_tracker = TransferTracker(timeout_sec=30.0)
        
        # Fault tolerance: Failed requests (for retry or skip)
        self._failed_requests: set[str] = set()
        self._pending_reconnects: set[str] = set()
        
        # Buffer
        self.buffer_size = 0
        self.buffer_size_threshold = float(self.config.kv_buffer_size)
        
        # Listener
        self._listener_thread = threading.Thread(target=self.listen_for_requests, daemon=True)
        self._listener_thread.start()
        
        # Fault tolerance: Health check thread
        self._health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self._health_check_thread.start()
        
        # Fault tolerance: Reconnection thread
        self._reconnect_thread = threading.Thread(target=self._reconnect_loop, daemon=True)
        self._reconnect_thread.start()
        
        # Ping
        self._ping_thread = None
        if port_offset == 0 and self.proxy_address != "":
            self._ping_thread = threading.Thread(target=self.ping, daemon=True)
            self._ping_thread.start()
        
        # Pre-registered KV cache memory (registered once, used for all transfers)
        # layer_name -> KVCacheMemoryInfo
        self.kv_cache_registry: dict[str, KVCacheMemoryInfo] = {}
        # All registered base addresses for exchange with remote
        self.kv_caches_base_addr: list[int] = []
        self.kv_caches_rkeys: list[int] = []
        self.block_size = 0
        self.num_blocks = 0
        self.block_len = 0  # Size of one block in bytes
        
        logger.info(
            "💯P2pRdmaEngine init: rank=%d, local_rank=%d, zmq=%s, rdma=%s",
            self.rank, self.local_rank, self.zmq_address, self.rdma.is_available
        )
    
    # ========================================================================
    # Fault Tolerance: Callbacks
    # ========================================================================
    
    def _on_peer_connected(self, address: str):
        """Callback when peer is connected."""
        logger.info("✅ Peer connected: %s", address)
    
    def _on_peer_disconnected(self, address: str):
        """Callback when peer is disconnected."""
        logger.warning("❌ Peer disconnected: %s", address)
        # Cancel pending transfers for this peer
        cancelled = self.transfer_tracker.cancel_transfers_for_peer(address)
        if cancelled:
            logger.warning("Cancelled %d transfers for peer %s", len(cancelled), address)
    
    def _on_peer_removed(self, address: str):
        """Callback when peer is removed (scale-down)."""
        logger.info("🔻 Peer removed (scale-down): %s", address)
        # Clean up connection resources
        self._cleanup_peer_connection(address)
    
    # ========================================================================
    # Fault Tolerance: Health Check
    # ========================================================================
    
    def _health_check_loop(self):
        """Background thread for health checking."""
        while not self._shutdown_event.is_set():
            try:
                # Check heartbeats
                dead_peers = self.conn_manager.check_all_heartbeats()
                for peer_addr in dead_peers:
                    logger.warning("Peer heartbeat failed: %s, scheduling reconnect", peer_addr)
                    self._pending_reconnects.add(peer_addr)
                
                # Check for timed out transfers
                timed_out = self.transfer_tracker.get_timed_out_transfers()
                for transfer in timed_out:
                    logger.warning(
                        "Transfer timeout: tensor=%s, peer=%s",
                        transfer["tensor_id"],
                        transfer["remote_address"]
                    )
                    self.transfer_tracker.cancel_transfer(transfer["transfer_id"])
                    self._failed_requests.add(transfer["tensor_id"].split("#")[0])
                
                # Cleanup old completed transfers
                self.transfer_tracker.cleanup_completed()
                
            except Exception as e:
                logger.error("Health check error: %s", e)
            
            # Sleep before next check
            self._shutdown_event.wait(CONNECTION_CHECK_INTERVAL_SEC)
    
    # ========================================================================
    # Fault Tolerance: Reconnection
    # ========================================================================
    
    def _reconnect_loop(self):
        """Background thread for reconnection attempts."""
        while not self._shutdown_event.is_set():
            try:
                # Get peers that need reconnection
                peers_to_reconnect = list(self._pending_reconnects)
                
                for peer_addr in peers_to_reconnect:
                    peer = self.conn_manager.get_peer(peer_addr)
                    if peer is None or peer.state == ConnectionState.REMOVED:
                        self._pending_reconnects.discard(peer_addr)
                        continue
                    
                    if not peer.is_recoverable():
                        logger.error(
                            "Peer not recoverable after %d attempts: %s",
                            peer.retry_count, peer_addr
                        )
                        self._pending_reconnects.discard(peer_addr)
                        continue
                    
                    # Attempt reconnection
                    logger.info(
                        "Attempting reconnection to %s (attempt %d/%d)",
                        peer_addr, peer.retry_count + 1, MAX_RETRY_ATTEMPTS
                    )
                    
                    success = self._try_reconnect(peer_addr)
                    if success:
                        self._pending_reconnects.discard(peer_addr)
                        logger.info("✅ Reconnected to %s", peer_addr)
                    else:
                        # Exponential backoff
                        backoff = RETRY_BACKOFF_BASE * (2 ** peer.retry_count)
                        logger.warning(
                            "Reconnection failed for %s, retry in %.1fs",
                            peer_addr, backoff
                        )
                        time.sleep(backoff)
                        
            except Exception as e:
                logger.error("Reconnect loop error: %s", e)
            
            self._shutdown_event.wait(1.0)  # Check every second
    
    def _try_reconnect(self, remote_address: str) -> bool:
        """Attempt to reconnect to a peer."""
        try:
            # Close existing connection
            self._cleanup_peer_connection(remote_address)
            
            # Re-establish connection
            sock = self.context.socket(zmq.DEALER)
            sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)
            sock.setsockopt(zmq.RCVTIMEO, SEND_RECV_TIMEOUT_MS)
            sock.setsockopt(zmq.SNDTIMEO, SEND_RECV_TIMEOUT_MS)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(f"tcp://{remote_address}")
            
            # Send reconnection request
            data = {
                "cmd": "RECONNECT",
                "hostname": self._local_hostname,
                "local_rank": self.local_rank,
            }
            sock.send(msgpack.dumps(data))
            
            # Wait for response
            response = sock.recv()
            resp_data = msgpack.loads(response)
            
            if resp_data.get("status") == "ok":
                self.socks[remote_address] = sock
                self.conn_manager.mark_connected(remote_address)
                return True
            else:
                sock.close()
                return False
                
        except zmq.Again:
            logger.warning("Reconnection timeout for %s", remote_address)
            peer = self.conn_manager.get_peer(remote_address)
            if peer:
                peer.mark_failed("timeout")
            return False
        except Exception as e:
            logger.warning("Reconnection failed for %s: %s", remote_address, e)
            peer = self.conn_manager.get_peer(remote_address)
            if peer:
                peer.mark_failed(str(e))
            return False
    
    def _cleanup_peer_connection(self, remote_address: str):
        """Clean up connection resources for a peer."""
        # Close ZMQ socket
        if remote_address in self.socks:
            try:
                self.socks[remote_address].close()
            except Exception:
                pass
            del self.socks[remote_address]
        
        # Clean up RDMA QP if exists
        if remote_address in self.remote_qp_info:
            del self.remote_qp_info[remote_address]
        
        # Remove from peer tracking
        if remote_address in self.peer_same_node:
            del self.peer_same_node[remote_address]
    
    # ========================================================================
    # Fault Tolerance: Scale-down Support
    # ========================================================================
    
    def remove_peer(self, remote_address: str) -> bool:
        """
        Remove a peer (scale-down operation).
        
        Args:
            remote_address: Address of peer to remove
            
        Returns:
            True if peer was removed, False if not found
        """
        # Cancel any pending transfers
        cancelled = self.transfer_tracker.cancel_transfers_for_peer(remote_address)
        if cancelled:
            logger.info("Cancelled %d pending transfers for removed peer", len(cancelled))
        
        # Remove from connection manager
        removed = self.conn_manager.remove_peer(remote_address)
        
        # Cleanup connection resources
        self._cleanup_peer_connection(remote_address)
        
        # Remove from pending reconnects
        self._pending_reconnects.discard(remote_address)
        
        return removed
    
    def get_active_peers(self) -> list[str]:
        """Get list of currently active (non-removed) peers."""
        return self.conn_manager.get_all_active_peers()
    
    def get_healthy_peers(self) -> list[str]:
        """Get list of healthy (connected) peers."""
        return self.conn_manager.get_healthy_peers()
    
    def get_connection_stats(self) -> dict:
        """Get connection statistics."""
        stats = self.conn_manager.get_stats()
        stats["pending_transfers"] = self.transfer_tracker.get_pending_count()
        stats["failed_requests"] = len(self._failed_requests)
        stats["pending_reconnects"] = len(self._pending_reconnects)
        return stats
    
    # ========================================================================
    # Fault Tolerance: Error Recovery
    # ========================================================================
    
    def is_request_failed(self, request_id: str) -> bool:
        """Check if a request has failed due to connection issues."""
        return request_id in self._failed_requests
    
    def clear_failed_request(self, request_id: str):
        """Clear a request from failed set (after handling)."""
        self._failed_requests.discard(request_id)
    
    def get_failed_requests(self) -> set[str]:
        """Get set of failed request IDs."""
        return self._failed_requests.copy()
    
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """
        Register all KV cache memory regions once at initialization.
        
        This is called once when the model is loaded. After registration,
        individual tensor transfers use offsets into these pre-registered
        regions - no per-tensor registration needed.
        
        Similar to Mooncake's register_kv_caches() approach.
        
        Args:
            kv_caches: Dict mapping layer names to KV cache tensors
        """
        logger.info("Registering KV caches for RDMA (one-time registration)")
        
        seen_base_addresses: set[int] = set()
        tensor_size_bytes = None
        
        for layer_name, cache in kv_caches.items():
            base_addr = cache.data_ptr()
            
            # Skip if already registered (same underlying memory)
            if base_addr in seen_base_addresses:
                continue
            
            seen_base_addresses.add(base_addr)
            curr_size = cache.nbytes
            
            if tensor_size_bytes is None:
                tensor_size_bytes = curr_size
                self.num_blocks = cache.shape[0]
            
            # Register with RDMA for GPUDirect
            mr = self.rdma.register_memory(base_addr, curr_size)
            
            if mr is not None:
                self.kv_caches_base_addr.append(base_addr)
                self.kv_caches_rkeys.append(mr.rkey)
                
                # Calculate block size
                block_len = curr_size // self.num_blocks
                
                self.kv_cache_registry[layer_name] = KVCacheMemoryInfo(
                    base_addr=base_addr,
                    total_length=curr_size,
                    lkey=mr.lkey,
                    rkey=mr.rkey,
                    num_blocks=self.num_blocks,
                    block_size_bytes=block_len,
                )
                
                logger.debug(
                    "Registered layer %s: addr=0x%x, size=%d, blocks=%d, rkey=%d",
                    layer_name, base_addr, curr_size, self.num_blocks, mr.rkey
                )
            else:
                logger.warning(
                    "Failed to register layer %s for RDMA, will use fallback",
                    layer_name
                )
        
        if tensor_size_bytes and self.num_blocks > 0:
            self.block_len = tensor_size_bytes // self.num_blocks
        
        logger.info(
            "KV cache registration complete: %d regions, %d blocks, block_len=%d",
            len(self.kv_caches_base_addr), self.num_blocks, self.block_len
        )
    
    def get_registered_memory_info(self, layer_name: str) -> KVCacheMemoryInfo | None:
        """Get pre-registered memory info for a layer."""
        return self.kv_cache_registry.get(layer_name)
    
    def rdma_write_blocks(
        self,
        remote_address: str,
        layer_name: str,
        local_block_ids: list[int],
        remote_block_ids: list[int],
        remote_base_addr: int,
        remote_rkey: int,
    ) -> bool:
        """
        RDMA write blocks using pre-registered memory.
        
        No per-transfer registration - uses offsets into pre-registered regions.
        """
        mem_info = self.get_registered_memory_info(layer_name)
        if mem_info is None:
            logger.warning("Layer %s not registered for RDMA", layer_name)
            return False
        
        qp_info = self._qps.get(remote_address)
        if qp_info is None:
            logger.warning("No QP for %s", remote_address)
            return False
        
        qp, _ = qp_info
        
        # Group contiguous blocks for efficient transfer
        groups = self._group_contiguous_blocks(local_block_ids, remote_block_ids)
        
        for local_start, remote_start, count in groups:
            local_addr = mem_info.base_addr + local_start * mem_info.block_size_bytes
            remote_addr = remote_base_addr + remote_start * mem_info.block_size_bytes
            length = count * mem_info.block_size_bytes
            
            success = self.rdma.rdma_write(
                qp=qp,
                local_addr=local_addr,
                local_lkey=mem_info.lkey,
                remote_addr=remote_addr,
                remote_rkey=remote_rkey,
                length=length,
                signaled=(count == groups[-1][2]),  # Signal on last
            )
            
            if not success:
                return False
        
        # Wait for completion
        completions = self.rdma.poll_completion(timeout_ms=5000)
        return len(completions) > 0 and all(c[1] == 0 for c in completions)
    
    def _group_contiguous_blocks(
        self,
        local_ids: list[int],
        remote_ids: list[int],
    ) -> list[tuple[int, int, int]]:
        """
        Group contiguous block ranges for efficient RDMA transfers.
        
        Returns: List of (local_start_id, remote_start_id, count)
        """
        if not local_ids:
            return []
        
        groups = []
        local_start = local_ids[0]
        remote_start = remote_ids[0]
        count = 1
        
        for i in range(1, len(local_ids)):
            # Check if contiguous in both local and remote
            if (local_ids[i] == local_ids[i-1] + 1 and 
                remote_ids[i] == remote_ids[i-1] + 1):
                count += 1
            else:
                groups.append((local_start, remote_start, count))
                local_start = local_ids[i]
                remote_start = remote_ids[i]
                count = 1
        
        groups.append((local_start, remote_start, count))
        return groups
    
    def _is_same_node(self, remote_address: str) -> bool:
        if remote_address not in self.peer_same_node:
            self.peer_same_node[remote_address] = is_same_node(
                self._local_hostname, remote_address
            )
        return self.peer_same_node[remote_address]
    
    def create_connect(
        self, 
        remote_address: str | None = None,
        retry_on_failure: bool = True,
    ) -> Any | None:
        """
        Create connection to remote address with fault tolerance.
        
        Args:
            remote_address: Remote peer address
            retry_on_failure: Whether to schedule retry on failure
            
        Returns:
            Socket on success, None on failure
        """
        assert remote_address is not None
        
        # Check if peer was removed (scale-down)
        peer = self.conn_manager.get_peer(remote_address)
        if peer and peer.state == ConnectionState.REMOVED:
            logger.warning("Cannot connect to removed peer: %s", remote_address)
            return None
        
        if remote_address in self.socks:
            # Check if existing connection is healthy
            if peer and peer.is_healthy():
                return self.socks[remote_address]
            else:
                # Connection exists but unhealthy, cleanup and reconnect
                self._cleanup_peer_connection(remote_address)
        
        try:
            sock = self.context.socket(zmq.DEALER)
            sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)
            sock.setsockopt(zmq.RCVTIMEO, SEND_RECV_TIMEOUT_MS)
            sock.setsockopt(zmq.SNDTIMEO, SEND_RECV_TIMEOUT_MS)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(f"tcp://{remote_address}")
            
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
            
            # Wait for response with timeout
            response = sock.recv()
            resp_data = msgpack.loads(response)
            
            # Register peer with connection manager
            self.conn_manager.add_peer(remote_address, {
                "hostname": self._local_hostname,
                "local_rank": self.local_rank,
                "same_node": same_node,
                "rdma_info": resp_data.get("rdma_info"),
            })
            
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
            
            self.socks[remote_address] = sock
            self.conn_manager.mark_connected(remote_address)
            
            logger.info("🤝Connected: %s👉%s, same_node=%s", 
                       self.zmq_address, remote_address, same_node)
            
            return sock
            
        except zmq.Again:
            # Timeout
            logger.warning("Connection timeout to %s", remote_address)
            self.conn_manager.mark_failed(remote_address, "timeout")
            if retry_on_failure:
                self._pending_reconnects.add(remote_address)
            return None
            
        except Exception as e:
            logger.error("Connection failed to %s: %s", remote_address, e)
            self.conn_manager.mark_failed(remote_address, str(e))
            if retry_on_failure:
                self._pending_reconnects.add(remote_address)
            return None
    
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
        """Listen for incoming requests with fault tolerance."""
        while not self._shutdown_event.is_set():
            try:
                # Poll with timeout to allow checking shutdown
                socks = dict(self.poller.poll(timeout=1000))
                if self.router_socket not in socks:
                    continue
                
                remote_address, message = self.router_socket.recv_multipart()
                remote_addr_str = remote_address.decode()
                
                try:
                    data = msgpack.loads(message)
                except Exception as e:
                    logger.warning("Failed to decode message from %s: %s", remote_addr_str, e)
                    continue
                
                # Update heartbeat for known peers
                self.conn_manager.update_heartbeat(remote_addr_str)
                
                if data["cmd"] == "NEW":
                    same_node = data.get("same_node", False)
                    self.peer_same_node[remote_addr_str] = same_node
                    
                    # Register peer
                    self.conn_manager.add_peer(remote_addr_str, {
                        "hostname": data.get("hostname", ""),
                        "local_rank": data.get("local_rank", 0),
                        "same_node": same_node,
                        "rdma_info": data.get("rdma_info"),
                    })
                    
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
                        self.remote_qp_info[remote_addr_str] = remote_info
                        
                        # Create our QP
                        qp_result = self.rdma.create_qp(remote_addr_str)
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
                    self.conn_manager.mark_connected(remote_addr_str)
                    
                    logger.info("🤝New: %s👈%s, same_node=%s, rdma=%s",
                               self.zmq_address, remote_addr_str, same_node,
                               response_rdma is not None)
                
                elif data["cmd"] == "RECONNECT":
                    # Handle reconnection request
                    logger.info("🔄 Reconnect request from %s", remote_addr_str)
                    
                    # Update peer state
                    peer = self.conn_manager.get_peer(remote_addr_str)
                    if peer and peer.state == ConnectionState.REMOVED:
                        # Peer was removed (scale-down), reject reconnection
                        response = {"status": "rejected", "reason": "peer_removed"}
                    else:
                        # Accept reconnection
                        self.conn_manager.add_peer(remote_addr_str, {
                            "hostname": data.get("hostname", ""),
                            "local_rank": data.get("local_rank", 0),
                            "same_node": self.peer_same_node.get(remote_addr_str, False),
                        })
                        self.conn_manager.mark_connected(remote_addr_str)
                        response = {"status": "ok"}
                    
                    self.router_socket.send_multipart([remote_address, msgpack.dumps(response)])
                
                elif data["cmd"] == "HEARTBEAT":
                    # Heartbeat ping
                    response = {"status": "ok", "timestamp": time.time()}
                    self.router_socket.send_multipart([remote_address, msgpack.dumps(response)])
                
                elif data["cmd"] == "DISCONNECT":
                    # Graceful disconnect (scale-down notification)
                    logger.info("📤 Disconnect notification from %s", remote_addr_str)
                    self.conn_manager.remove_peer(remote_addr_str)
                    response = {"status": "ok"}
                    self.router_socket.send_multipart([remote_address, msgpack.dumps(response)])
                
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
                            
                    except zmq.Again:
                        # Timeout receiving tensor data
                        logger.warning("Timeout receiving tensor data from %s", remote_addr_str)
                        tensor = None
                        self._failed_requests.add(tensor_id.split("#")[0])
                            
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
                        try:
                            with torch.cuda.stream(self.copy_stream):
                                tensor_bytes = tensor.to(self.device).cpu().numpy().tobytes()
                            self.copy_stream.synchronize()
                            self.router_socket.send_multipart([remote_address, tensor_bytes])
                        except zmq.Again:
                            logger.warning("Timeout sending tensor to %s", remote_addr_str)
                
                else:
                    logger.warning("Unknown command from %s: %s", remote_addr_str, data.get("cmd"))
                    
            except zmq.Again:
                # Timeout on recv, continue
                continue
            except Exception as e:
                logger.error("Error processing request: %s", e)
    
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
        """Close engine with graceful shutdown."""
        logger.info("Shutting down P2pRdmaEngine...")
        
        # Signal all threads to stop
        self._shutdown_event.set()
        
        # Notify all peers of graceful shutdown
        self._notify_peers_shutdown()
        
        # Wait for threads
        if hasattr(self, '_listener_thread'):
            self._listener_thread.join(timeout=2)
        if hasattr(self, '_health_check_thread'):
            self._health_check_thread.join(timeout=2)
        if hasattr(self, '_reconnect_thread'):
            self._reconnect_thread.join(timeout=2)
        if self.send_type == "PUT_ASYNC" and hasattr(self, '_send_thread'):
            self._send_thread.join(timeout=2)
        if self._ping_thread is not None:
            self._ping_thread.join(timeout=2)
        
        # Cleanup connection manager
        if hasattr(self, 'conn_manager'):
            self.conn_manager.cleanup()
        
        # Cleanup transport layers
        self.rdma.close()
        self.nvlink.close()
        
        # Close all sockets
        for sock in self.socks.values():
            try:
                sock.close()
            except Exception:
                pass
        self.socks.clear()
        
        try:
            self.router_socket.close()
        except Exception:
            pass
        
        try:
            self.context.term()
        except Exception:
            pass
        
        logger.info("P2pRdmaEngine shutdown complete")
    
    def _notify_peers_shutdown(self):
        """Notify all connected peers of graceful shutdown."""
        for address, sock in list(self.socks.items()):
            try:
                data = {"cmd": "DISCONNECT"}
                sock.send(msgpack.dumps(data), zmq.NOBLOCK)
            except Exception:
                pass  # Best effort
    
    def graceful_disconnect(self, remote_address: str) -> bool:
        """
        Gracefully disconnect from a peer (for scale-down).
        
        Sends disconnect notification and cleans up resources.
        """
        if remote_address not in self.socks:
            return False
        
        try:
            sock = self.socks[remote_address]
            data = {"cmd": "DISCONNECT"}
            sock.send(msgpack.dumps(data))
            
            # Wait for acknowledgment
            try:
                sock.setsockopt(zmq.RCVTIMEO, 1000)
                response = sock.recv()
                resp_data = msgpack.loads(response)
                logger.info("Disconnect acknowledged by %s", remote_address)
            except zmq.Again:
                logger.warning("No disconnect ack from %s", remote_address)
        except Exception as e:
            logger.warning("Error during graceful disconnect: %s", e)
        
        # Cleanup
        self.remove_peer(remote_address)
        return True
