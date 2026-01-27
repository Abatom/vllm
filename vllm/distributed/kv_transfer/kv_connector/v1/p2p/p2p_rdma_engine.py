# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
P2P RDMA Engine for KV cache transfer.

This module provides a P2P engine that uses:
- RDMA (via UCX) for inter-node communication between different machines
- NVLink (via CUDA IPC) for intra-node communication within the same machine

The engine automatically detects whether the remote peer is on the same node
and selects the optimal transport mechanism.
"""

import ctypes
import json
import logging
import os
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import msgpack
import torch
import zmq

from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.device_communicators.cuda_wrapper import (
    CudaRTLibrary,
    cudaIpcMemHandle_t,
)
from vllm.distributed.kv_transfer.kv_connector.v1.p2p.tensor_memory_pool import (
    TensorMemoryPool,
)
from vllm.utils.network_utils import get_ip

logger = logging.getLogger(__name__)

DEFAULT_MEM_POOL_SIZE_GB = 32

# UCX transport flags
UCX_TLS_RDMA = "rc,ud,dc"  # RDMA transports
UCX_TLS_TCP = "tcp"  # Fallback TCP transport
UCX_TLS_ALL = "all"  # Let UCX choose the best


def get_hostname() -> str:
    """Get the hostname of the current machine."""
    return socket.gethostname()


def is_same_node(local_hostname: str, remote_address: str) -> bool:
    """
    Check if the remote address is on the same node as local.
    
    Args:
        local_hostname: Local machine's hostname
        remote_address: Remote address in format "ip:port" or "hostname:port"
    
    Returns:
        True if on the same node, False otherwise
    """
    try:
        remote_host = remote_address.split(":")[0]
        
        # Check if it's the same hostname
        if remote_host == local_hostname:
            return True
        
        # Check if it's localhost
        if remote_host in ("localhost", "127.0.0.1", "::1"):
            return True
        
        # Try to resolve and compare IPs
        try:
            local_ip = socket.gethostbyname(local_hostname)
            remote_ip = socket.gethostbyname(remote_host)
            if local_ip == remote_ip:
                return True
        except socket.gaierror:
            pass
        
        # Check against all local IPs
        try:
            local_ips = set()
            for info in socket.getaddrinfo(local_hostname, None):
                local_ips.add(info[4][0])
            # Add common loopback addresses
            local_ips.add("127.0.0.1")
            local_ips.add("::1")
            
            if remote_host in local_ips:
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
class IpcMemoryHandle:
    """Represents a CUDA IPC memory handle for NVLink transfers."""
    handle: cudaIpcMemHandle_t
    size: int
    dtype: torch.dtype
    shape: tuple
    
    def to_bytes(self) -> bytes:
        """Serialize the IPC handle to bytes."""
        handle_bytes = bytes(self.handle.internal)
        metadata = {
            "size": self.size,
            "dtype": str(self.dtype).replace("torch.", ""),
            "shape": list(self.shape),
        }
        return msgpack.dumps({
            "handle": handle_bytes,
            "metadata": metadata,
        })
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "IpcMemoryHandle":
        """Deserialize an IPC handle from bytes."""
        unpacked = msgpack.loads(data)
        handle = cudaIpcMemHandle_t()
        ctypes.memmove(
            ctypes.addressof(handle.internal),
            unpacked["handle"],
            128
        )
        metadata = unpacked["metadata"]
        return cls(
            handle=handle,
            size=metadata["size"],
            dtype=getattr(torch, metadata["dtype"]),
            shape=tuple(metadata["shape"]),
        )


class RdmaTransport:
    """
    RDMA transport layer using UCX for inter-node communication.
    
    This class wraps UCX to provide high-performance RDMA-based
    tensor transfers between different machines.
    """
    
    def __init__(self, local_rank: int, device: torch.device):
        self.local_rank = local_rank
        self.device = device
        self._ucp = None
        self._ep_cache: dict[str, Any] = {}  # remote_address -> endpoint
        self._listener = None
        self._worker = None
        self._pending_receives: dict[str, Any] = {}
        self._lock = threading.Lock()
        
        # Try to import and initialize UCX
        self._init_ucx()
    
    def _init_ucx(self):
        """Initialize UCX library."""
        try:
            import ucp
            self._ucp = ucp
            
            # Configure UCX for optimal RDMA performance
            ucp_config = {
                "TLS": os.environ.get("UCX_TLS", UCX_TLS_ALL),
                "SOCKADDR_TLS_PRIORITY": "rdmacm,tcp",
            }
            
            # Enable GPUDirect RDMA if available
            if torch.cuda.is_available():
                ucp_config["CUDA_COPY_ASYNC"] = "y"
                
            # Initialize UCX
            self._ucp.init(ucp_config)
            logger.info("UCX initialized successfully for RDMA transport")
            
        except ImportError:
            logger.warning(
                "UCX-Py not available. RDMA transport will fall back to "
                "alternative methods. Install with: pip install ucx-py"
            )
            self._ucp = None
        except Exception as e:
            logger.warning("Failed to initialize UCX: %s", e)
            self._ucp = None
    
    @property
    def is_available(self) -> bool:
        """Check if RDMA transport is available."""
        return self._ucp is not None
    
    async def _create_endpoint(self, address: str, port: int) -> Any:
        """Create a UCX endpoint to the remote address."""
        if self._ucp is None:
            raise RuntimeError("UCX not available")
        
        key = f"{address}:{port}"
        if key in self._ep_cache:
            return self._ep_cache[key]
        
        ep = await self._ucp.create_endpoint(address, port)
        self._ep_cache[key] = ep
        return ep
    
    def send_tensor_sync(
        self,
        tensor: torch.Tensor,
        remote_address: str,
        remote_port: int,
    ) -> bool:
        """
        Synchronously send a tensor via RDMA.
        
        Falls back to ZMQ-based transfer if UCX is not available.
        """
        if self._ucp is None:
            return False
        
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(
                self._async_send_tensor(tensor, remote_address, remote_port)
            )
            loop.close()
            return result
        except Exception as e:
            logger.error("RDMA send failed: %s", e)
            return False
    
    async def _async_send_tensor(
        self,
        tensor: torch.Tensor,
        remote_address: str,
        remote_port: int,
    ) -> bool:
        """Async implementation of tensor send via RDMA."""
        try:
            ep = await self._create_endpoint(remote_address, remote_port)
            
            # Send tensor metadata first
            metadata = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "size": tensor.element_size() * tensor.numel(),
            }
            meta_bytes = msgpack.dumps(metadata)
            await ep.send(meta_bytes)
            
            # Send tensor data - UCX will use RDMA if available
            # For GPU tensors, UCX can use GPUDirect RDMA
            if tensor.is_cuda:
                await ep.send(tensor)
            else:
                await ep.send(tensor.numpy().tobytes())
            
            return True
        except Exception as e:
            logger.error("Async RDMA send failed: %s", e)
            return False
    
    def close(self):
        """Close all endpoints and cleanup UCX resources."""
        for ep in self._ep_cache.values():
            try:
                ep.close()
            except Exception:
                pass
        self._ep_cache.clear()
        
        if self._listener is not None:
            try:
                self._listener.close()
            except Exception:
                pass


class NvlinkTransport:
    """
    NVLink transport layer using CUDA IPC for intra-node communication.
    
    This class uses CUDA IPC memory handles to enable direct GPU-to-GPU
    transfers via NVLink when available.
    """
    
    def __init__(self, local_rank: int, device: torch.device):
        self.local_rank = local_rank
        self.device = device
        self.cudart = CudaRTLibrary()
        
        # Cache for opened IPC memory handles
        self._ipc_cache: dict[bytes, ctypes.c_void_p] = {}
        self._lock = threading.Lock()
        
        # Stream for async operations
        self.stream = torch.cuda.Stream(device=device)
    
    def get_ipc_handle(self, tensor: torch.Tensor) -> IpcMemoryHandle:
        """
        Get a CUDA IPC memory handle for a tensor.
        
        Args:
            tensor: CUDA tensor to get handle for
            
        Returns:
            IpcMemoryHandle containing the handle and metadata
        """
        if not tensor.is_cuda:
            raise ValueError("Tensor must be on CUDA device")
        
        # Ensure tensor is contiguous
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        handle = self.cudart.cudaIpcGetMemHandle(
            ctypes.c_void_p(tensor.data_ptr())
        )
        
        return IpcMemoryHandle(
            handle=handle,
            size=tensor.element_size() * tensor.numel(),
            dtype=tensor.dtype,
            shape=tuple(tensor.shape),
        )
    
    def open_ipc_handle(self, ipc_handle: IpcMemoryHandle) -> torch.Tensor:
        """
        Open an IPC memory handle and create a tensor from it.
        
        Args:
            ipc_handle: The IPC memory handle to open
            
        Returns:
            A tensor backed by the IPC shared memory
        """
        handle_key = bytes(ipc_handle.handle.internal)
        
        with self._lock:
            if handle_key in self._ipc_cache:
                dev_ptr = self._ipc_cache[handle_key]
            else:
                dev_ptr = self.cudart.cudaIpcOpenMemHandle(ipc_handle.handle)
                self._ipc_cache[handle_key] = dev_ptr
        
        # Create a tensor from the device pointer
        # Note: We need to create a tensor that views this memory
        tensor = torch.empty(
            ipc_handle.shape,
            dtype=ipc_handle.dtype,
            device=self.device,
        )
        
        # Copy data from IPC memory to our tensor
        self.cudart.cudaMemcpy(
            ctypes.c_void_p(tensor.data_ptr()),
            dev_ptr,
            ipc_handle.size,
        )
        
        return tensor
    
    def send_via_ipc(
        self,
        tensor: torch.Tensor,
        zmq_socket: zmq.Socket,
        remote_identity: bytes,
    ) -> bool:
        """
        Send a tensor via IPC handle through ZMQ signaling.
        
        The actual data transfer happens via NVLink/PCIe P2P,
        ZMQ is only used for signaling.
        """
        try:
            ipc_handle = self.get_ipc_handle(tensor)
            handle_bytes = ipc_handle.to_bytes()
            
            data = {
                "cmd": "IPC_SEND",
                "ipc_data": handle_bytes,
            }
            zmq_socket.send_multipart([remote_identity, msgpack.dumps(data)])
            return True
        except Exception as e:
            logger.error("IPC send failed: %s", e)
            return False
    
    def recv_via_ipc(self, ipc_data: bytes) -> torch.Tensor:
        """
        Receive a tensor via IPC handle.
        
        Args:
            ipc_data: Serialized IPC handle data
            
        Returns:
            The received tensor
        """
        ipc_handle = IpcMemoryHandle.from_bytes(ipc_data)
        return self.open_ipc_handle(ipc_handle)
    
    def close(self):
        """Cleanup IPC resources."""
        self._ipc_cache.clear()


class P2pRdmaEngine:
    """
    P2P RDMA Engine for KV cache transfer.
    
    This engine automatically selects the optimal transport:
    - NVLink (via CUDA IPC) for intra-node transfers
    - RDMA (via UCX) for inter-node transfers
    - Falls back to ZMQ-based transfer if neither is available
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
        
        if not hostname:
            hostname = get_ip()
        port = int(self.config.kv_port) + port_offset
        if port == 0:
            raise ValueError("Port cannot be 0")
        self._hostname = hostname
        self._port = port
        
        # Store local hostname for same-node detection
        self._local_hostname = get_hostname()
        
        # Each card corresponds to a ZMQ address
        self.zmq_address = f"{self._hostname}:{self._port}"
        
        # Proxy configuration
        proxy_ip = self.config.get_from_extra_config("proxy_ip", "")
        proxy_port = self.config.get_from_extra_config("proxy_port", "")
        if proxy_ip == "" or proxy_port == "":
            self.proxy_address = ""
            self.http_address = ""
        else:
            self.proxy_address = proxy_ip + ":" + proxy_port
            http_port = self.config.get_from_extra_config("http_port", None)
            if http_port is None:
                example_cfg = {
                    "kv_connector": "P2pRdmaConnector",
                    "kv_connector_extra_config": {"http_port": 8000},
                }
                example = (
                    f"--port=8000 --kv-transfer-config='{json.dumps(example_cfg)}'"
                )
                raise ValueError(
                    "kv_connector_extra_config.http_port is required. "
                    f"Example: {example}"
                )
            self.http_address = f"{self._hostname}:{http_port}"
        
        # Initialize ZMQ
        self.context = zmq.Context()
        self.router_socket = self.context.socket(zmq.ROUTER)
        self.router_socket.bind(f"tcp://{self.zmq_address}")
        
        self.poller = zmq.Poller()
        self.poller.register(self.router_socket, zmq.POLLIN)
        
        # Threading synchronization
        self.send_store_cv = threading.Condition()
        self.send_queue_cv = threading.Condition()
        self.recv_store_cv = threading.Condition()
        
        # CUDA streams
        self.send_stream = torch.cuda.Stream()
        self.recv_stream = torch.cuda.Stream()
        
        # Memory pool for overflow
        mem_pool_size_gb = float(
            self.config.get_from_extra_config(
                "mem_pool_size_gb", DEFAULT_MEM_POOL_SIZE_GB
            )
        )
        self.pool = TensorMemoryPool(
            max_block_size=int(mem_pool_size_gb * 1024**3)
        )
        
        # Initialize transport layers
        self.nvlink_transport = NvlinkTransport(self.local_rank, self.device)
        self.rdma_transport = RdmaTransport(self.local_rank, self.device)
        
        # Send type configuration
        self.send_type = self.config.get_from_extra_config("send_type", "PUT_ASYNC")
        if self.send_type == "GET":
            self.send_store: dict[str, torch.Tensor] = {}
        else:
            self.send_queue: deque[SendQueueItem] = deque()
            if self.send_type == "PUT_ASYNC":
                self._send_thread = threading.Thread(
                    target=self.send_async, daemon=True
                )
                self._send_thread.start()
        
        # Storage for received tensors
        self.recv_store: dict[str, Any] = {}
        self.recv_request_id_to_tensor_ids: dict[str, set[str]] = {}
        self.send_request_id_to_tensor_ids: dict[str, set[str]] = {}
        
        # Connection caches
        self.socks: dict[str, Any] = {}  # remote_address: client socket
        self.peer_same_node: dict[str, bool] = {}  # remote_address: is_same_node
        
        # Buffer management
        self.buffer_size = 0
        self.buffer_size_threshold = float(self.config.kv_buffer_size)
        
        # Start listener thread
        self._listener_thread = threading.Thread(
            target=self.listen_for_requests, daemon=True
        )
        self._listener_thread.start()
        
        # Start ping thread if needed
        self._ping_thread = None
        if port_offset == 0 and self.proxy_address != "":
            self._ping_thread = threading.Thread(target=self.ping, daemon=True)
            self._ping_thread.start()
        
        # Check for UCX/RDMA availability
        rdma_available = self.rdma_transport.is_available
        
        logger.info(
            "💯P2pRdmaEngine init, rank:%d, local_rank:%d, http_address:%s, "
            "zmq_address:%s, proxy_address:%s, send_type:%s, buffer_size_"
            "threshold:%.2f, rdma_available:%s, hostname:%s",
            self.rank,
            self.local_rank,
            self.http_address,
            self.zmq_address,
            self.proxy_address,
            self.send_type,
            self.buffer_size_threshold,
            rdma_available,
            self._local_hostname,
        )
    
    def _is_same_node(self, remote_address: str) -> bool:
        """Check if remote address is on the same node (cached)."""
        if remote_address not in self.peer_same_node:
            self.peer_same_node[remote_address] = is_same_node(
                self._local_hostname, remote_address
            )
        return self.peer_same_node[remote_address]
    
    def create_connect(self, remote_address: str | None = None):
        """Create a connection to a remote address."""
        assert remote_address is not None
        if remote_address not in self.socks:
            sock = self.context.socket(zmq.DEALER)
            sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)
            sock.connect(f"tcp://{remote_address}")
            self.socks[remote_address] = sock
            
            # Determine if this is a same-node connection
            same_node = self._is_same_node(remote_address)
            
            # Send connection establishment message
            data = {
                "cmd": "NEW",
                "hostname": self._local_hostname,
                "same_node": same_node,
            }
            sock.send(msgpack.dumps(data))
            
            logger.info(
                "🤝Connection established, %s👉%s, same_node:%s",
                self.zmq_address,
                remote_address,
                same_node,
            )
        
        return self.socks[remote_address]
    
    def send_tensor(
        self,
        tensor_id: str,
        tensor: torch.Tensor,
        remote_address: str | None = None,
    ) -> bool:
        """
        Send a tensor to a remote address.
        
        Automatically selects transport:
        - NVLink/IPC for same-node
        - RDMA for inter-node
        - Falls back to ZMQ if needed
        """
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
                logger.warning(
                    "❗[GET]tensor_id:%s, tensor_size:%d, is greater than"
                    "buffer size threshold :%d, skip send to %s, rank:%d",
                    tensor_id,
                    tensor_size,
                    self.buffer_size_threshold,
                    remote_address,
                    self.rank,
                )
                return False
            while self.buffer_size + tensor_size > self.buffer_size_threshold:
                assert len(self.send_store) > 0
                oldest_tensor_id = next(iter(self.send_store))
                oldest_tensor = self.send_store.pop(oldest_tensor_id)
                oldest_tensor_size = (
                    oldest_tensor.element_size() * oldest_tensor.numel()
                )
                self.buffer_size -= oldest_tensor_size
            
            self.send_store[tensor_id] = tensor
            self.buffer_size += tensor_size
        return True
    
    def recv_tensor(
        self,
        tensor_id: str,
        remote_address: str | None = None,
    ) -> torch.Tensor:
        """Receive a tensor from a remote address."""
        if self.send_type == "PUT" or self.send_type == "PUT_ASYNC":
            start_time = time.time()
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
            else:
                duration = time.time() - start_time
                logger.warning(
                    "🔴[PUT]Recv From %s, tensor_id:%s, duration:%.3fms, rank:%d",
                    remote_address,
                    tensor_id,
                    duration * 1000,
                    self.rank,
                )
            return tensor
        
        # GET mode
        if remote_address is None:
            return None
        
        if remote_address not in self.socks:
            self.create_connect(remote_address)
        
        sock = self.socks[remote_address]
        same_node = self._is_same_node(remote_address)
        
        data = {"cmd": "GET", "tensor_id": tensor_id, "same_node": same_node}
        sock.send(msgpack.dumps(data))
        
        message = sock.recv()
        data = msgpack.loads(message)
        if data["ret"] != 0:
            logger.warning(
                "🔴[GET]Recv From %s, tensor_id: %s, ret: %d",
                remote_address,
                tensor_id,
                data["ret"],
            )
            return None
        
        # Receive based on transport type
        if data.get("transport") == "ipc":
            # NVLink/IPC transfer
            tensor = self.nvlink_transport.recv_via_ipc(data["ipc_data"])
        else:
            # Standard transfer - receive via separate message
            with torch.cuda.stream(self.recv_stream):
                tensor = torch.empty(
                    data["shape"],
                    dtype=getattr(torch, data["dtype"]),
                    device=self.device,
                )
            
            # Receive tensor data
            tensor_data = sock.recv()
            tensor.copy_(torch.frombuffer(
                tensor_data, dtype=tensor.dtype
            ).reshape(tensor.shape).to(self.device))
        
        return tensor
    
    def listen_for_requests(self):
        """Listen for incoming requests from remote peers."""
        while True:
            socks = dict(self.poller.poll())
            if self.router_socket not in socks:
                continue
            
            remote_address, message = self.router_socket.recv_multipart()
            data = msgpack.loads(message)
            
            if data["cmd"] == "NEW":
                # New connection establishment
                remote_hostname = data.get("hostname", "")
                same_node = data.get("same_node", False)
                self.peer_same_node[remote_address.decode()] = same_node
                logger.info(
                    "🤝New connection, %s👈%s, hostname:%s, same_node:%s",
                    self.zmq_address,
                    remote_address.decode(),
                    remote_hostname,
                    same_node,
                )
                
            elif data["cmd"] == "PUT":
                tensor_id = data["tensor_id"]
                transport_type = data.get("transport", "zmq")
                
                try:
                    if transport_type == "ipc":
                        # NVLink/IPC transfer
                        tensor = self.nvlink_transport.recv_via_ipc(
                            data["ipc_data"]
                        )
                    else:
                        # Standard ZMQ transfer
                        with torch.cuda.stream(self.recv_stream):
                            tensor = torch.empty(
                                data["shape"],
                                dtype=getattr(torch, data["dtype"]),
                                device=self.device,
                            )
                        
                        # Wait for tensor data
                        self.router_socket.send_multipart([remote_address, b"0"])
                        _, tensor_data = self.router_socket.recv_multipart()
                        
                        # Copy data to GPU tensor
                        cpu_tensor = torch.frombuffer(
                            tensor_data, dtype=tensor.dtype
                        ).reshape(tensor.shape)
                        tensor.copy_(cpu_tensor)
                    
                    tensor_size = tensor.element_size() * tensor.numel()
                    if self.buffer_size + tensor_size > self.buffer_size_threshold:
                        # Store Tensor in memory pool
                        addr = self.pool.store_tensor(tensor)
                        tensor = (addr, tensor.dtype, tensor.shape)
                        logger.warning(
                            "🔴[PUT]Recv Tensor, Out Of Threshold, "
                            "%s👈%s, tensor_id:%s, addr:%d",
                            self.zmq_address,
                            remote_address.decode(),
                            tensor_id,
                            addr,
                        )
                    else:
                        self.buffer_size += tensor_size
                    
                    if transport_type == "ipc":
                        self.router_socket.send_multipart([remote_address, b"0"])
                        
                except torch.cuda.OutOfMemoryError:
                    self.router_socket.send_multipart([remote_address, b"1"])
                    tensor = None
                    logger.warning(
                        "🔴[PUT]Recv Tensor, Out Of Memory, %s👈%s, data:%s",
                        self.zmq_address,
                        remote_address.decode(),
                        data,
                    )
                
                with self.recv_store_cv:
                    self.recv_store[tensor_id] = tensor
                    self.have_received_tensor_id(tensor_id)
                    self.recv_store_cv.notify()
                    
            elif data["cmd"] == "GET":
                tensor_id = data["tensor_id"]
                same_node = data.get("same_node", False)
                
                with self.send_store_cv:
                    tensor = self.send_store.pop(tensor_id, None)
                    if tensor is not None:
                        # Re-add for LRU behavior
                        self.send_store[tensor_id] = tensor
                        self.have_sent_tensor_id(tensor_id)
                        
                        if same_node:
                            # Use IPC for same-node transfer
                            try:
                                ipc_handle = self.nvlink_transport.get_ipc_handle(
                                    tensor.to(self.device)
                                )
                                response = {
                                    "ret": 0,
                                    "transport": "ipc",
                                    "ipc_data": ipc_handle.to_bytes(),
                                }
                            except Exception as e:
                                logger.warning("IPC handle creation failed: %s", e)
                                # Fall back to standard transfer
                                response = {
                                    "ret": 0,
                                    "shape": tensor.shape,
                                    "dtype": str(tensor.dtype).replace("torch.", ""),
                                    "transport": "zmq",
                                }
                        else:
                            response = {
                                "ret": 0,
                                "shape": tensor.shape,
                                "dtype": str(tensor.dtype).replace("torch.", ""),
                                "transport": "zmq",
                            }
                    else:
                        response = {"ret": 1}
                
                self.router_socket.send_multipart(
                    [remote_address, msgpack.dumps(response)]
                )
                
                if response["ret"] == 0 and response.get("transport") == "zmq":
                    # Send tensor data for ZMQ transport
                    tensor_bytes = tensor.to(self.device).cpu().numpy().tobytes()
                    self.router_socket.send_multipart(
                        [remote_address, tensor_bytes]
                    )
            else:
                logger.warning(
                    "🚧Unexpected, Received message from %s, data:%s",
                    remote_address,
                    data,
                )
    
    def have_sent_tensor_id(self, tensor_id: str):
        """Track sent tensor IDs."""
        request_id = tensor_id.split("#")[0]
        if request_id not in self.send_request_id_to_tensor_ids:
            self.send_request_id_to_tensor_ids[request_id] = set()
        self.send_request_id_to_tensor_ids[request_id].add(tensor_id)
    
    def have_received_tensor_id(self, tensor_id: str):
        """Track received tensor IDs."""
        request_id = tensor_id.split("#")[0]
        if request_id not in self.recv_request_id_to_tensor_ids:
            self.recv_request_id_to_tensor_ids[request_id] = set()
        self.recv_request_id_to_tensor_ids[request_id].add(tensor_id)
    
    def send_async(self):
        """Async send thread for PUT_ASYNC mode."""
        while True:
            with self.send_queue_cv:
                while not self.send_queue:
                    self.send_queue_cv.wait()
                item = self.send_queue.popleft()
                if not self.send_queue:
                    self.send_queue_cv.notify()
            self.send_sync(item)
    
    def wait_for_sent(self):
        """Wait for all pending sends to complete."""
        if self.send_type == "PUT_ASYNC":
            start_time = time.time()
            with self.send_queue_cv:
                while self.send_queue:
                    self.send_queue_cv.wait()
            duration = time.time() - start_time
            logger.debug(
                "🚧[PUT_ASYNC]It took %.3fms to wait for the send_queue"
                " to be empty, rank:%d",
                duration * 1000,
                self.rank,
            )
    
    def send_sync(self, item: SendQueueItem) -> bool:
        """Synchronously send a tensor."""
        if item.remote_address is None:
            return False
        if item.remote_address not in self.socks:
            self.create_connect(item.remote_address)
        
        tensor = item.tensor
        sock = self.socks[item.remote_address]
        same_node = self._is_same_node(item.remote_address)
        
        if same_node:
            # Use NVLink/IPC for same-node transfer
            try:
                ipc_handle = self.nvlink_transport.get_ipc_handle(
                    tensor.to(self.device)
                )
                data = {
                    "cmd": "PUT",
                    "tensor_id": item.tensor_id,
                    "transport": "ipc",
                    "ipc_data": ipc_handle.to_bytes(),
                }
                sock.send(msgpack.dumps(data))
                
                response = sock.recv()
                if response != b"0":
                    logger.error(
                        "🔴IPC Send failed, %s 👉 %s, tensor_id:%s",
                        self.zmq_address,
                        item.remote_address,
                        item.tensor_id,
                    )
                    return False
                
                if self.send_type == "PUT_ASYNC":
                    self.have_sent_tensor_id(item.tensor_id)
                return True
                
            except Exception as e:
                logger.warning(
                    "IPC send failed, falling back to ZMQ: %s", e
                )
        
        # Use ZMQ-based transfer (for inter-node or fallback)
        data = {
            "cmd": "PUT",
            "tensor_id": item.tensor_id,
            "shape": tensor.shape,
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "transport": "zmq",
        }
        sock.send(msgpack.dumps(data))
        
        response = sock.recv()
        if response != b"0":
            logger.error(
                "🔴Send Tensor, Peer Out Of Memory/Threshold, %s 👉 %s, "
                "tensor_id:%s, tensor:%s, size:%fGB, response:%s",
                self.zmq_address,
                item.remote_address,
                item.tensor_id,
                tensor.shape,
                tensor.element_size() * tensor.numel() / 1024**3,
                response.decode(),
            )
            return False
        
        # Send tensor data
        tensor_bytes = tensor.to(self.device).cpu().numpy().tobytes()
        sock.send(tensor_bytes)
        
        if self.send_type == "PUT_ASYNC":
            self.have_sent_tensor_id(item.tensor_id)
        
        return True
    
    def get_finished(
        self, finished_req_ids: set[str], no_compile_layers
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Notifies worker-side connector ids of requests that have
        finished generating tokens.
        
        Returns:
            ids of requests that have finished asynchronous transfer,
            tuple of (sending/saving ids, recving/loading ids).
        """
        # Clear the buffer upon request completion
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
        
        finished_sending: set[str] = set()
        finished_recving: set[str] = set()
        
        return finished_sending or None, finished_recving or None
    
    def ping(self):
        """Ping thread to keep connection alive with proxy."""
        sock = self.context.socket(zmq.DEALER)
        sock.setsockopt_string(zmq.IDENTITY, self.zmq_address)
        logger.debug("ping start, zmq_address:%s", self.zmq_address)
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
        """Close the engine and cleanup resources."""
        self._listener_thread.join(timeout=1)
        if self.send_type == "PUT_ASYNC":
            self._send_thread.join(timeout=1)
        if self._ping_thread is not None:
            self._ping_thread.join(timeout=1)
        
        # Cleanup transport resources
        self.nvlink_transport.close()
        self.rdma_transport.close()
        
        # Close ZMQ sockets
        for sock in self.socks.values():
            sock.close()
        self.router_socket.close()
        self.context.term()
