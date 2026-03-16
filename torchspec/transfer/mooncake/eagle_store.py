# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import atexit
import ctypes
import hashlib
import os
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from torchspec.transfer.mooncake.deferred_delete import DeferredDeleteManager
from torchspec.transfer.mooncake.helpers import _format_bytes
from torchspec.transfer.mooncake.store import MooncakeHiddenStateStore
from torchspec.utils.logging import logger

if TYPE_CHECKING:
    from torchspec.models.target.eagle3_target_model import Eagle3TargetOutput

# Static lookup for dtype → element size in bytes (avoids creating a tensor
# on every call to _compute_tensor_size).
_DTYPE_ELEMENT_SIZES = {
    torch.float64: 8,
    torch.float32: 4,
    torch.bfloat16: 2,
    torch.float16: 2,
    torch.int64: 8,
    torch.int32: 4,
    torch.int16: 2,
    torch.int8: 1,
    torch.uint8: 1,
    torch.bool: 1,
}


class EagleMooncakeStore(MooncakeHiddenStateStore):
    """
    Mooncake Store wrapper specialized for Eagle3 hidden states.

    Uses RDMA-registered host buffers and put_from for zero-copy transfers.
    Each Eagle3 output is stored as multiple tensors with key suffixes:
    - {key}_hs: hidden_states
    - {key}_tgt: target
    - {key}_ids: input_ids
    - {key}_lhs: last_hidden_states (if present)

    Deletions are deferred to respect Mooncake's lease TTL (config.kv_lease_ttl_s).
    """

    TENSOR_SUFFIXES = ["_hs", "_tgt", "_ids", "_lhs"]

    def __init__(self, config):
        """Initialize Eagle3 Mooncake Store with deferred deletion."""
        super().__init__(config)
        self._deferred_delete_manager: Optional[DeferredDeleteManager] = None
        self._cleanup_registered = False

    def setup(self, device: torch.device = None) -> None:
        """Initialize the Mooncake Store client and deferred delete manager."""
        super().setup(device)

        if self._deferred_delete_manager is None:
            lease_ttl_s = self.config.kv_lease_ttl_s
            # Initialize deferred delete manager after store is ready
            self._deferred_delete_manager = DeferredDeleteManager(
                store=self._store,
                ttl_buffer_seconds=0.5,  # Small buffer for safety
                check_interval=1.0,  # Check queue every second
                max_queue_size=10000,  # Max pending deletions
                retry_interval=2.0,  # Retry failed deletes after 2s
                ttl_seconds=lease_ttl_s,  # Mooncake lease TTL
            )
            logger.debug("Deferred delete manager initialized")

            # Register cleanup on exit
            if not self._cleanup_registered:
                atexit.register(self._cleanup_deferred_deletes)
                self._cleanup_registered = True

    def _cleanup_deferred_deletes(self):
        """Cleanup deferred delete manager on exit."""
        if self._deferred_delete_manager is not None:
            logger.info("Cleaning up deferred delete manager...")
            stats = self._deferred_delete_manager.get_stats()
            queue_size = self._deferred_delete_manager.get_queue_size()
            if queue_size > 0:
                logger.warning(
                    " Shutting down with %d pending deletions. "
                    "Some Mooncake objects may not be cleaned up.",
                    queue_size,
                )
            self._deferred_delete_manager.stop()
            logger.info(
                "Deferred delete final stats: %s",
                stats,
            )

    def _secondary_storage_enabled(self) -> bool:
        return bool(getattr(self.config, "secondary_storage_dir", None))

    def _disk_path_for_key(self, key: str) -> str:
        if not self._secondary_storage_enabled():
            raise RuntimeError("secondary storage is not configured")
        root = os.path.abspath(self.config.secondary_storage_dir)
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return os.path.join(root, f"{digest}.pt")

    def _disk_exists(self, key: str) -> bool:
        if not self._secondary_storage_enabled():
            return False
        return os.path.exists(self._disk_path_for_key(key))

    def _has_complete_disk_bundle(self, keys: List[str]) -> bool:
        return bool(keys) and all(self._disk_exists(key) for key in keys)

    def _spill_entries_to_disk(self, entries: List[Tuple[str, torch.Tensor, int]]) -> None:
        if not self._secondary_storage_enabled():
            raise RuntimeError("secondary storage is not configured")
        root = os.path.abspath(self.config.secondary_storage_dir)
        os.makedirs(root, exist_ok=True)
        for tensor_key, tensor, _ in entries:
            path = self._disk_path_for_key(tensor_key)
            tmp_path = path + ".tmp"
            torch.save(tensor.detach().cpu(), tmp_path)
            os.replace(tmp_path, path)

    def _get_tensors_from_disk(
        self,
        keys: List[str],
        tensor_specs: List[Tuple[str, Tuple[int, ...], torch.dtype]],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        tensor_map: Dict[str, torch.Tensor] = {}
        for key, (name, shape, dtype) in zip(keys, tensor_specs):
            path = self._disk_path_for_key(key)
            tensor = torch.load(path, map_location="cpu", weights_only=False)
            if tuple(tensor.shape) != tuple(shape):
                raise RuntimeError(
                    f"Disk spill shape mismatch for {name}: got {tuple(tensor.shape)}, expected {shape}"
                )
            tensor = tensor.to(dtype=dtype)
            tensor_map[name] = tensor.to(device)
            if name == "input_ids":
                tensor_map["input_ids_cpu"] = tensor.clone()
        return tensor_map

    def remove(self, key: str) -> None:
        disk_removed = False
        if self._disk_exists(key):
            os.remove(self._disk_path_for_key(key))
            disk_removed = True
        try:
            super().remove(key)
        except Exception:
            if not disk_removed:
                raise

    def exists(self, key: str) -> bool:
        return self._disk_exists(key) or super().exists(key)

    def put(
        self,
        key: str,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        last_hidden_states: Optional[torch.Tensor],
        target: Optional[torch.Tensor] = None,
    ) -> Dict[str, Tuple[int, ...]]:
        """Store Eagle3 output tensors via async batch_put_from.

        DtoH staging runs on ``_copy_stream`` so the caller's compute stream
        is never blocked.  The RDMA transfer runs on a background thread via
        ``AsyncPutManager``.  With *pool_size* host buffers the caller almost
        never waits — ``wait_for_buffer`` only blocks when every buffer is
        still in-flight.

        For GPU Direct send the path is synchronous (no DtoH needed).
        """
        self._ensure_initialized()
        logger.debug("put: starting for key=%s", key)
        entries = [(f"{key}_hs", hidden_states), (f"{key}_ids", input_ids)]

        if target is not None:
            entries.append((f"{key}_tgt", target))

        if last_hidden_states is not None:
            entries.append((f"{key}_lhs", last_hidden_states))

        non_empty_entries = []
        total_bytes = 0
        for tensor_key, tensor in entries:
            nbytes = tensor.element_size() * tensor.numel()
            total_bytes += nbytes
            if nbytes > 0:
                non_empty_entries.append((tensor_key, tensor, nbytes))

        keys = [tensor_key for tensor_key, _, _ in non_empty_entries]
        sizes = [nbytes for _, _, nbytes in non_empty_entries]
        buffer_ptrs = []

        shapes = {
            "hidden_states": tuple(hidden_states.shape),
            "input_ids": tuple(input_ids.shape),
        }
        if target is not None:
            shapes["target"] = tuple(target.shape)
        if last_hidden_states is not None:
            shapes["last_hidden_states"] = tuple(last_hidden_states.shape)

        if (
            non_empty_entries
            and self._secondary_storage_enabled()
            and getattr(self.config, "spill_to_disk_on_failure", False)
            and total_bytes > self.config.host_buffer_size
        ):
            logger.warning(
                "Mooncake put for key=%s exceeds host buffer (%s > %s); spilling %d tensors to secondary storage (%s).",
                key,
                _format_bytes(total_bytes),
                _format_bytes(self.config.host_buffer_size),
                len(non_empty_entries),
                self.config.secondary_storage_dir,
            )
            self._spill_entries_to_disk(non_empty_entries)
            return shapes

        host_buf = None
        if non_empty_entries:
            if self._host_buffer_pool is None:
                # Trainer-side stores may be configured with async_put_pool_size=0.
                # Fall back to a single synchronous host buffer so local chunk writes
                # still work when the store is primarily used for get().
                from torchspec.transfer.mooncake.buffers import HostBufferPool

                self._host_buffer_pool = HostBufferPool(
                    buffer_size=self.config.host_buffer_size,
                    pool_size=1,
                )
                self._host_buffer_pool.initialize()
                for buf in self._host_buffer_pool._buffers:
                    self._register_buffer(buf.ptr, buf.size)
                logger.info(
                    "Lazily initialized host buffer pool for synchronous put fallback: %.1fMB",
                    self.config.host_buffer_size / (1024**2),
                )
            host_buf = self._host_buffer_pool.get_buffer()
            offset = 0
            try:
                for _, tensor, nbytes in non_empty_entries:
                    copied = host_buf.copy_from_tensor(tensor, offset=offset)
                    if copied != nbytes:
                        raise RuntimeError(
                            f"Unexpected staged byte count for Mooncake key {key}: copied={copied}, expected={nbytes}"
                        )
                    buffer_ptrs.append(host_buf.ptr + offset)
                    offset += nbytes
            except ValueError:
                if self._secondary_storage_enabled() and getattr(
                    self.config, "spill_to_disk_on_failure", False
                ):
                    logger.warning(
                        "Mooncake staging overflow for key=%s; spilling %d tensors to secondary storage (%s).",
                        key,
                        len(non_empty_entries),
                        self.config.secondary_storage_dir,
                    )
                    self._spill_entries_to_disk(non_empty_entries)
                    return shapes
                raise

        def _run_batch_put() -> List[Tuple[str, int, int, int]]:
            if not keys:
                return []
            results = self._store.batch_put_from(keys, buffer_ptrs, sizes)
            failures: List[Tuple[str, int, int, int]] = []
            for k, r, ptr, size in zip(keys, results, buffer_ptrs, sizes):
                if r != 0:
                    failures.append((k, r, ptr, size))
            return failures

        failures = _run_batch_put()
        if failures:
            for k in keys:
                try:
                    self._store.remove(k)
                except Exception:
                    logger.debug(
                        "Failed to remove partial key %s after batch_put_from failure.",
                        k,
                    )
            failure_details = ", ".join(
                f"{k} (code={r}, ptr={ptr}, size={size})" for k, r, ptr, size in failures
            )
            config_details = (
                f"total_bytes={_format_bytes(total_bytes)}, "
                f"global_segment_size={_format_bytes(self.config.global_segment_size)}, "
                f"local_buffer_size={_format_bytes(self.config.local_buffer_size)}, "
                f"host_buffer_size={_format_bytes(self.config.host_buffer_size)}"
            )
            if self._secondary_storage_enabled() and getattr(self.config, "spill_to_disk_on_failure", False):
                logger.warning(
                    "Mooncake put failed for key=%s; spilling %d tensors to secondary storage (%s). failures=%s",
                    key,
                    len(non_empty_entries),
                    self.config.secondary_storage_dir,
                    failure_details,
                )
                self._spill_entries_to_disk(non_empty_entries)
                return shapes
            raise RuntimeError(
                f"batch_put_from failed for keys: {failure_details}. "
                f"{config_details}. Consider increasing Mooncake segment/buffer sizes "
                "or reducing batch/sequence length/prefetch depth."
            )

        logger.debug(
            "[MOONCAKE] put: completed key=%s, total_bytes=%s, shapes=%s",
            key,
            _format_bytes(total_bytes),
            shapes,
        )
        return shapes

    @staticmethod
    def _stage_tensors_into_buffer(buf, tensors: List[torch.Tensor]) -> Tuple[List[int], List[int]]:
        """Copy tensors into a buffer and return (pointers, sizes) for batch_put_from."""
        buffer_ptrs = []
        sizes = []
        offset = 0
        for tensor in tensors:
            nbytes = buf.copy_from_tensor(tensor, offset=offset)
            buffer_ptrs.append(buf.ptr + offset)
            sizes.append(nbytes)
            offset += nbytes
        return buffer_ptrs, sizes

    def get(
        self,
        key: str,
        shapes: Dict[str, Tuple[int, ...]],
        dtypes: Dict[str, torch.dtype],
        device: torch.device,
    ) -> "Eagle3TargetOutput":
        """
        Retrieve Eagle3 tensors into GPU memory.

        For RDMA/InfiniBand: Uses GPUDirect RDMA (batch_get_into directly into GPU).
        For TCP: Uses batch_get_buffer to host buffer, then copies to GPU.

        Automatically falls back to host buffer path if GPUDirect fails.

        Returns:
            Eagle3TargetOutput with the retrieved tensors.
        """
        self._ensure_initialized()

        from torchspec.models.target.eagle3_target_model import Eagle3TargetOutput

        tensor_specs = [
            (
                "hidden_states",
                shapes["hidden_states"],
                dtypes.get("hidden_states", torch.bfloat16),
                f"{key}_hs",
            ),
            ("input_ids", shapes["input_ids"], torch.int64, f"{key}_ids"),
        ]

        if "target" in shapes:
            tensor_specs.append(
                ("target", shapes["target"], dtypes.get("target", torch.bfloat16), f"{key}_tgt")
            )

        if "last_hidden_states" in shapes:
            tensor_specs.append(
                (
                    "last_hidden_states",
                    shapes["last_hidden_states"],
                    dtypes.get("hidden_states", torch.bfloat16),
                    f"{key}_lhs",
                )
            )

        fetch_keys = []
        fetch_specs = []
        zero_specs = []
        for name, shape, dtype, tensor_key in tensor_specs:
            if self._compute_tensor_size(shape, dtype) == 0:
                zero_specs.append((name, shape, dtype))
            else:
                fetch_keys.append(tensor_key)
                fetch_specs.append((name, shape, dtype))

        tensor_map = {}
        if fetch_keys:
            if self._has_complete_disk_bundle(fetch_keys):
                tensor_map = self._get_tensors_from_disk(fetch_keys, fetch_specs, device)
                logger.debug("Using secondary disk storage path")
            else:
                tensor_map = None
                if self._gpu_direct_available and self._gpu_receive_buffer is not None:
                    tensor_map = self._get_tensors_gpu_direct(fetch_keys, fetch_specs, device)
                    if tensor_map is None:
                        logger.warning("GPUDirect batch_get_into failed; falling back to host buffer path.")

                if tensor_map is None:
                    tensor_map = self._get_tensors_via_host_buffer(fetch_keys, fetch_specs, device)
                    logger.debug("Using host buffer path (TCP)")

        for name, shape, dtype in zero_specs:
            tensor_map[name] = torch.empty(shape, dtype=dtype, device=device)
            if name == "input_ids":
                tensor_map["input_ids_cpu"] = torch.empty(shape, dtype=dtype, device="cpu")

        logger.debug("Retrieved Eagle3 tensors with base key: %s", key)

        return Eagle3TargetOutput(
            hidden_states=tensor_map["hidden_states"],
            target=tensor_map.get("target"),
            input_ids=tensor_map["input_ids"],
            last_hidden_states=tensor_map.get("last_hidden_states"),
            input_ids_cpu=tensor_map.get("input_ids_cpu"),
        )

    def _get_tensors_gpu_direct(
        self,
        keys: List[str],
        tensor_specs: List[Tuple[str, Tuple[int, ...], torch.dtype]],
        device: torch.device,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Transfer directly into GPU memory using batch_get_into (GPUDirect RDMA).

        Pre-allocates individual destination tensors and transfers directly into
        their memory, avoiding an extra GPU→GPU copy.  Falls back to the
        registered GPU receive buffer when a tensor is too small to be
        page-aligned (RDMA typically requires registered memory).

        Returns None if transfer fails, allowing caller to fall back to host
        buffer path.
        """
        total_size = sum(
            self._compute_tensor_size(shape, dtype) for _, shape, dtype in tensor_specs
        )

        if total_size > self._gpu_receive_buffer.size:
            logger.warning(
                "GPU buffer too small: need %.1fMB, have %.1fMB. Increase gpu_buffer_size in config.",
                total_size / (1024**2),
                self._gpu_receive_buffer.size / (1024**2),
            )
            return None

        # Compute per-tensor sizes and buffer offsets up-front.
        buffer_ptrs: List[int] = []
        sizes: List[int] = []
        offsets: List[int] = []
        offset = 0

        for _, shape, dtype in tensor_specs:
            size = self._compute_tensor_size(shape, dtype)
            buffer_ptrs.append(self._gpu_receive_buffer.ptr + offset)
            sizes.append(size)
            offsets.append(offset)
            offset += size

        try:
            results = self._store.batch_get_into(keys, buffer_ptrs, sizes)
            for i, (k, r) in enumerate(zip(keys, results)):
                if r < 0:
                    logger.warning("batch_get_into failed for %s with error code: %s", k, r)
                    return None
                if r != 0 and r != sizes[i]:
                    logger.warning(
                        "batch_get_into for %s: unexpected return %s (expected 0 or %s)",
                        k,
                        r,
                        sizes[i],
                    )
        except Exception as e:
            logger.warning("batch_get_into exception: %s", e)
            return None

        tensor_map = {}
        for i, (name, shape, dtype) in enumerate(tensor_specs):
            numel = 1
            for dim in shape:
                numel *= dim
            buf_slice = self._gpu_receive_buffer.get_slice(offsets[i], sizes[i])
            # View into the registered buffer; valid until the next call.
            tensor_map[name] = buf_slice.view(dtype)[:numel].reshape(shape)

        logger.debug("GPU Direct RDMA transfer successful for %s tensors", len(keys))
        return tensor_map

    @staticmethod
    def _compute_tensor_size(shape: Tuple[int, ...], dtype: torch.dtype) -> int:
        """Compute the byte size of a tensor with given shape and dtype."""
        numel = 1
        for dim in shape:
            numel *= dim
        return numel * _DTYPE_ELEMENT_SIZES[dtype]

    def _get_tensors_via_host_buffer(
        self,
        keys: List[str],
        tensor_specs: List[Tuple[str, Tuple[int, ...], torch.dtype]],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Transfer via Mooncake's registered host buffer, then copy to device."""
        wait_seconds = max(self.config.get_retry_wait_seconds, 0.05)
        log_interval = max(self.config.get_retry_log_interval_seconds, wait_seconds)
        max_wait = max(self.config.get_retry_max_wait_seconds, 0.0)
        start_time = time.time()
        last_log = 0.0

        while True:
            buffers = self._store.batch_get_buffer(keys)
            missing = [i for i, buf in enumerate(buffers) if buf is None]
            if not missing:
                break

            elapsed = time.time() - start_time
            if max_wait > 0 and elapsed >= max_wait:
                missing_keys = ", ".join(keys[i] for i in missing)
                raise RuntimeError(
                    f"batch_get_buffer returned None for keys: {missing_keys}. "
                    f"Waited {elapsed:.1f}s; aborting."
                )

            now = time.time()
            if last_log == 0.0 or (now - last_log) >= log_interval:
                missing_keys = ", ".join(keys[i] for i in missing)
                logger.warning(
                    "batch_get_buffer missing keys (%s); sleeping %.2fs.",
                    missing_keys,
                    wait_seconds,
                )
                last_log = now
            time.sleep(wait_seconds)

        tensor_map = {}
        for i, ((name, shape, dtype), buf) in enumerate(zip(tensor_specs, buffers)):
            if buf is None:
                raise RuntimeError(
                    f"batch_get_buffer returned None for key '{keys[i]}' (tensor: {name}). "
                    "This may indicate the key doesn't exist or RDMA transfer failed."
                )

            numel = 1
            for dim in shape:
                numel *= dim
            element_size = _DTYPE_ELEMENT_SIZES[dtype]
            expected_size = numel * element_size

            buf_size = buf.size()
            if buf_size != expected_size:
                actual_elements = buf_size // element_size if element_size > 0 else 0
                logger.error(
                    f"Size mismatch for {name} (key={keys[i]}): "
                    f"got {buf_size} bytes ({actual_elements} elements), "
                    f"expected {expected_size} bytes ({numel} elements). "
                    f"Expected shape: {shape}, dtype: {dtype}, element_size: {element_size}"
                )
                raise RuntimeError(
                    f"Size mismatch for {name}: got {buf_size} bytes, expected {expected_size} bytes"
                )

            c_array = (ctypes.c_byte * buf_size).from_address(buf.ptr())
            host_tensor = torch.frombuffer(c_array, dtype=dtype, count=numel).reshape(shape)

            tensor_map[name] = host_tensor.to(device)

            if name == "input_ids":
                tensor_map["input_ids_cpu"] = host_tensor.clone()

        return tensor_map

    def remove_eagle3_tensors(
        self,
        key: str,
        has_last_hidden_states: bool = False,
        has_target: bool = False,
    ) -> None:
        """
        Queue deferred removal of all tensors associated with an Eagle3 output.

        Deletions are queued and executed after Mooncake's lease TTL expires.
        This prevents deletion failures due to active leases.

        Args:
            key: Base key used when storing
            has_last_hidden_states: Whether last_hidden_states was stored
            has_target: Whether target (logits) was stored
        """

        keys = [f"{key}_hs", f"{key}_ids"]
        if has_target:
            keys.append(f"{key}_tgt")
        if has_last_hidden_states:
            keys.append(f"{key}_lhs")

        for tensor_key in keys:
            if self._disk_exists(tensor_key):
                try:
                    os.remove(self._disk_path_for_key(tensor_key))
                except FileNotFoundError:
                    pass

        mooncake_keys = [tensor_key for tensor_key in keys if super(EagleMooncakeStore, self).exists(tensor_key)]
        if not mooncake_keys:
            return

        logger.debug(
            "Queueing deferred deletion for base_key=%s, num_keys=%d",
            key,
            len(mooncake_keys),
        )

        if self._deferred_delete_manager is None:
            logger.error(
                "Deferred delete manager not initialized! Cannot delete %s",
                key,
            )
            return

        success = self._deferred_delete_manager.enqueue_delete(
            keys=mooncake_keys,
            base_key=key,
            max_attempts=3,
        )

        if success:
            logger.debug(
                "Queued deferred deletion for base_key=%s",
                key,
            )
        else:
            logger.error(
                "Failed to queue deletion for %s (queue full)",
                key,
            )

    def get_deferred_delete_stats(self) -> Dict[str, int]:
        """Get statistics from the deferred delete manager.

        Returns:
            Dict with keys: enqueued, attempted, succeeded, failed, retried, abandoned, queue_size
        """
        if self._deferred_delete_manager is None:
            return {
                "enqueued": 0,
                "attempted": 0,
                "succeeded": 0,
                "failed": 0,
                "retried": 0,
                "abandoned": 0,
                "queue_size": 0,
            }

        stats = self._deferred_delete_manager.get_stats()
        stats["queue_size"] = self._deferred_delete_manager.get_queue_size()
        return stats
