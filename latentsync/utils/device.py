# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Device utilities for cross-platform support (CUDA, Apple Silicon MPS, CPU)."""

import contextlib
import os
from typing import Optional, Union

import torch


def is_mps_available() -> bool:
    """Check if Apple Silicon MPS backend is available."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built()


def get_device(prefer: Optional[str] = None) -> torch.device:
    """Return the best available device.

    Order of preference: CUDA -> MPS -> CPU. Honors the LATENTSYNC_DEVICE env
    variable, and an optional `prefer` argument ("cuda", "mps", "cpu", or
    "cuda:N") when that device is actually available.
    """
    env_device = os.environ.get("LATENTSYNC_DEVICE")
    if env_device:
        prefer = env_device

    if prefer:
        prefer_lower = prefer.lower()
        if prefer_lower.startswith("cuda") and torch.cuda.is_available():
            return torch.device(prefer)
        if prefer_lower == "mps" and is_mps_available():
            return torch.device("mps")
        if prefer_lower == "cpu":
            return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if is_mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_str(prefer: Optional[str] = None) -> str:
    """Return the device as a string (e.g. "cuda", "mps", "cpu")."""
    return str(get_device(prefer))


def get_autocast_dtype(device: Optional[Union[str, torch.device]] = None) -> torch.dtype:
    """Return a dtype suitable for inference on the given device.

    CUDA with compute capability >= 8 gets float16, otherwise float32. MPS
    technically supports float16 but it is unstable for diffusion workloads,
    so we default to float32. CPU always uses float32.
    """
    if device is None:
        device = get_device()
    device = torch.device(device) if not isinstance(device, torch.device) else device

    if device.type == "cuda":
        try:
            major = torch.cuda.get_device_capability(device.index or 0)[0]
            if major >= 8:
                return torch.float16
        except Exception:
            pass
        return torch.float16
    return torch.float32


def supports_fp16(device: Optional[Union[str, torch.device]] = None) -> bool:
    """Return True if fp16 ops are reliable on this device."""
    if device is None:
        device = get_device()
    device = torch.device(device) if not isinstance(device, torch.device) else device
    return device.type == "cuda"


def empty_cache(device: Optional[Union[str, torch.device]] = None) -> None:
    """Free unused cached memory for the active accelerator. Safe no-op on CPU."""
    if device is None:
        device = get_device()
    device = torch.device(device) if not isinstance(device, torch.device) else device

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == "mps" and is_mps_available():
        try:
            torch.mps.empty_cache()
        except AttributeError:
            pass


def device_count(device: Optional[Union[str, torch.device]] = None) -> int:
    """Return the number of accelerator devices.

    MPS exposes a single virtual device. CPU returns 0 since there is no
    parallel GPU resource to spread workers across.
    """
    if device is None:
        device = get_device()
    device = torch.device(device) if not isinstance(device, torch.device) else device

    if device.type == "cuda":
        return torch.cuda.device_count()
    if device.type == "mps":
        return 1
    return 0


def manual_seed_all(seed: int) -> None:
    """Seed the active accelerator. Safe no-op on backends that aren't initialized."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if is_mps_available():
        try:
            torch.mps.manual_seed(seed)
        except AttributeError:
            pass


def set_device(device: Union[int, str, torch.device]) -> None:
    """Set the current device. No-op for MPS/CPU since they aren't indexed."""
    if isinstance(device, int):
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
        return

    device = torch.device(device) if not isinstance(device, torch.device) else device
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(device.index or 0)


@contextlib.contextmanager
def autocast(device: Optional[Union[str, torch.device]] = None, dtype: Optional[torch.dtype] = None, enabled: bool = True):
    """Backend-aware autocast context.

    Falls back to a no-op on devices that don't support autocast for the given
    dtype (e.g. MPS with fp16 is unreliable for diffusion).
    """
    if device is None:
        device = get_device()
    device = torch.device(device) if not isinstance(device, torch.device) else device

    if not enabled or device.type == "cpu":
        yield
        return

    if device.type == "cuda":
        amp_dtype = dtype if dtype is not None else torch.float16
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
            yield
        return

    # MPS: skip autocast — diffusion stacks hit unsupported ops in fp16/bf16.
    yield


def make_grad_scaler(device: Optional[Union[str, torch.device]] = None, enabled: bool = True):
    """Return a GradScaler only when the device benefits from one (CUDA)."""
    if not enabled:
        return None
    if device is None:
        device = get_device()
    device = torch.device(device) if not isinstance(device, torch.device) else device

    if device.type == "cuda":
        return torch.amp.GradScaler("cuda")
    return None


def distributed_backend(device: Optional[Union[str, torch.device]] = None) -> str:
    """Return the appropriate torch.distributed backend for the device."""
    if device is None:
        device = get_device()
    device = torch.device(device) if not isinstance(device, torch.device) else device

    if device.type == "cuda":
        return "nccl"
    return "gloo"


def onnx_providers(device: Optional[Union[str, torch.device]] = None):
    """Return a list of ONNX Runtime execution providers for the device."""
    if device is None:
        device = get_device()
    device = torch.device(device) if not isinstance(device, torch.device) else device

    if device.type == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device.type == "mps":
        # CoreML is the closest ONNX equivalent on Apple Silicon, but it isn't
        # always available; fall back to CPU.
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def device_to_ctx_id(device: Union[str, torch.device]) -> int:
    """Convert a device into an integer ctx_id for libraries like InsightFace.

    -1 means CPU. CUDA devices use the device index. MPS is treated as -1 since
    InsightFace's onnxruntime backend has no MPS provider.
    """
    if isinstance(device, str):
        if device.lower() == "cpu":
            return -1
        device = torch.device(device)

    if device.type == "cuda":
        return device.index if device.index is not None else 0
    return -1
