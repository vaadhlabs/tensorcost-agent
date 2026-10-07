"""
MIG inspector (spec 4C) — read-only visibility of NVIDIA MIG mode,
partition layout, and per-process GPU memory usage.

This module never mutates MIG state; it only reports. Where NVML is not
available or MIG isn't supported (older drivers / non-MIG devices), every
function returns a benign empty / False shape so backend code can assume
the same contract on every machine. Hopper (A100/H100/H200) and Blackwell
(B200 / GB200 GPU) support MIG — see profile tables below.
"""

import logging
from typing import Dict, List, Optional, Tuple

from src import gpu_model_key

logger = logging.getLogger(__name__)

try:
    import pynvml
    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False


def is_mig_supported() -> bool:
    """Coarse check — NVML is present and can be initialized."""
    if not _NVML_AVAILABLE:
        return False
    try:
        pynvml.nvmlInit()
        return True
    except Exception:
        return False


def get_mig_mode(handle) -> Dict[str, object]:
    """
    Return the current + pending MIG-mode state for a device handle.

    Output: { current_enabled: bool, pending_enabled: bool }
    On any failure, returns {current_enabled: False, pending_enabled: False}.
    """
    out = {'current_enabled': False, 'pending_enabled': False}
    if not _NVML_AVAILABLE or not device_supports_mig(handle):
        return out
    try:
        current, pending = pynvml.nvmlDeviceGetMigMode(handle)
        # NVML returns ints where 1 = ENABLED, 0 = DISABLED.
        out['current_enabled'] = bool(current)
        out['pending_enabled'] = bool(pending)
    except Exception as exc:
        logger.debug(f"get_mig_mode failed: {exc}")
    return out


def get_mig_partitions(handle) -> List[Dict[str, object]]:
    """
    Return the list of MIG device instances currently attached to the GPU.
    Each entry carries {index, uuid, memory_total_mb, memory_used_mb}.
    Returns [] on non-MIG devices or any NVML error.
    """
    if not _NVML_AVAILABLE or not device_supports_mig(handle):
        return []
    partitions: List[Dict[str, object]] = []
    try:
        # nvmlDeviceGetMaxMigDeviceCount returns the max number of MIG device
        # slots; iterate and ask for the handle of each. Slots without an
        # active instance raise NVMLError — we swallow and skip.
        max_instances = pynvml.nvmlDeviceGetMaxMigDeviceCount(handle)
        for i in range(max_instances):
            try:
                mig_handle = pynvml.nvmlDeviceGetMigDeviceHandleByIndex(handle, i)
            except Exception:
                continue
            try:
                uuid = pynvml.nvmlDeviceGetUUID(mig_handle)
            except Exception:
                uuid = None
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(mig_handle)
                mem_total = round(mem.total / 1024 / 1024, 2)
                mem_used = round(mem.used / 1024 / 1024, 2)
            except Exception:
                mem_total = None
                mem_used = None
            partitions.append({
                'index': i,
                'uuid': uuid.decode() if isinstance(uuid, bytes) else uuid,
                'memory_total_mb': mem_total,
                'memory_used_mb': mem_used,
                # GI slice count breaks memory ties (1g.45gb vs 2g.45gb).
                'gi_slice': _mig_gi_slice(mig_handle),
            })
    except Exception as exc:
        logger.debug(f"get_mig_partitions failed: {exc}")
    return partitions


def get_running_processes(handle) -> List[Dict[str, object]]:
    """
    Return the list of compute processes currently using this GPU.
    Each entry: {pid, used_memory_mb}. Empty on error.
    """
    if not _NVML_AVAILABLE:
        return []
    procs: List[Dict[str, object]] = []
    try:
        # Compute + graphics processes; ignore missing symbol on older drivers.
        getters = []
        if hasattr(pynvml, 'nvmlDeviceGetComputeRunningProcesses'):
            getters.append(pynvml.nvmlDeviceGetComputeRunningProcesses)
        if hasattr(pynvml, 'nvmlDeviceGetGraphicsRunningProcesses'):
            getters.append(pynvml.nvmlDeviceGetGraphicsRunningProcesses)
        for fn in getters:
            try:
                for p in fn(handle):
                    procs.append({
                        'pid': p.pid,
                        'used_memory_mb': round(getattr(p, 'usedGpuMemory', 0) / 1024 / 1024, 2)
                                           if getattr(p, 'usedGpuMemory', None) else None
                    })
            except Exception as exc:
                logger.debug(f"process enumeration failed: {exc}")
    except Exception as exc:
        logger.debug(f"get_running_processes failed: {exc}")
    return procs


# Profile tables — memory_gb + compute_slice match gpu-service seed convention.
# H100/H200 use expanded Hopper profiles (1g.10gb/1g.18gb, etc.).
_MIG_PROFILES_A100: List[Tuple[str, int, int]] = [
    ("1g.5gb", 5, 14),
    ("2g.10gb", 10, 28),
    ("3g.20gb", 20, 42),
    ("7g.40gb", 40, 100),
]
_MIG_PROFILES_H100: List[Tuple[str, int, int]] = [
    ("1g.10gb", 10, 14),
    ("1g.20gb", 20, 14),
    ("2g.20gb", 20, 28),
    ("3g.40gb", 40, 42),
    ("4g.40gb", 40, 57),
    ("7g.80gb", 80, 100),
]
_MIG_PROFILES_H200: List[Tuple[str, int, int]] = [
    ("1g.18gb", 18, 14),
    ("1g.35gb", 35, 14),
    ("2g.35gb", 35, 28),
    ("3g.71gb", 71, 42),
    ("4g.71gb", 71, 57),
    ("7g.141gb", 141, 100),
]
# NVIDIA MIG User Guide / gpu-operator mig-parted configs.
# B200 and GB200 share 1g.23gb but diverge above that (45/90/180 vs 47/93/186).
_MIG_PROFILES_B200: List[Tuple[str, int, int]] = [
    ("1g.23gb", 23, 14),
    ("1g.45gb", 45, 14),
    ("2g.45gb", 45, 28),
    ("3g.90gb", 90, 42),
    ("4g.90gb", 90, 57),
    ("7g.180gb", 180, 100),
]
# GB200 HGX device-filter profiles (gpu-operator all-balanced + GB200 configs).
# Omits 1g.24gb / 3g.95gb / 4g.95gb / 7g.189gb — those appear under H100-96GB
# and GH200 blocks in the same configmap, not the GB200 device-filter.
_MIG_PROFILES_GB200: List[Tuple[str, int, int]] = [
    ("1g.23gb", 23, 14),
    ("1g.47gb", 47, 14),
    ("2g.47gb", 47, 28),
    ("3g.93gb", 93, 42),
    ("4g.93gb", 93, 57),
    ("7g.186gb", 186, 100),
]
_MIG_PROFILES_BY_FAMILY = {
    "A100": _MIG_PROFILES_A100,
    "H100": _MIG_PROFILES_H100,
    "H200": _MIG_PROFILES_H200,
    "B200": _MIG_PROFILES_B200,
    "GB200": _MIG_PROFILES_GB200,
}


def _device_name(handle) -> Optional[str]:
    if not _NVML_AVAILABLE:
        return None
    try:
        raw = pynvml.nvmlDeviceGetName(handle)
        return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    except Exception:
        return None


def device_supports_mig(handle) -> bool:
    """Return False for non-MIG GPUs without raising."""
    name = _device_name(handle)
    if isinstance(name, str) and name.strip():
        return gpu_model_key.is_mig_capable(name)
    return True


def _profile_gi_slice(profile: str) -> Optional[int]:
    """Parse the Ng prefix from a profile name like '4g.90gb'."""
    if not profile or "g." not in profile:
        return None
    head = profile.split("g.", 1)[0]
    return int(head) if head.isdigit() else None


def _mig_gi_slice(mig_handle) -> Optional[int]:
    """
    Read GPU-instance slice count for a MIG device handle.

    Prefer parsing the MIG device name (NVML reports names like
    "NVIDIA B200 MIG 4g.90gb"). Falling back to profile structs must use
    nvmlDeviceGetGpuInstanceProfileInfoById(V) — ProfileInfo(V) takes a
    profile *index*, not the profileId from GpuInstanceInfo.
    """
    if not _NVML_AVAILABLE:
        return None
    # 1. Name parse — works without walking parent GI profile tables.
    name = _device_name(mig_handle)
    if isinstance(name, str):
        parsed = _profile_gi_slice_from_device_name(name)
        if parsed is not None:
            return parsed
    # 2. Profile-id lookup (ById / ByIdV), never ProfileInfo(index).
    try:
        parent = pynvml.nvmlDeviceGetDeviceHandleFromMigDeviceHandle(mig_handle)
        gi_id = pynvml.nvmlDeviceGetGpuInstanceId(mig_handle)
        gi = pynvml.nvmlDeviceGetGpuInstanceById(parent, gi_id)
        info = pynvml.nvmlGpuInstanceGetInfo(gi)
        by_id = getattr(pynvml, "nvmlDeviceGetGpuInstanceProfileInfoByIdV", None)
        if by_id is None:
            by_id = getattr(pynvml, "nvmlDeviceGetGpuInstanceProfileInfoById", None)
        if by_id is None:
            return None
        prof = by_id(parent, info.profileId)
        raw_name = getattr(prof, "name", None)
        if raw_name:
            pname = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
            parsed = _profile_gi_slice(pname.replace("MIG ", "").strip())
            if parsed is not None:
                return parsed
        sc = getattr(prof, "sliceCount", None)
        if sc is not None:
            return int(sc)
    except Exception:
        return None
    return None


def _profile_gi_slice_from_device_name(name: str) -> Optional[int]:
    """Extract Ng from an NVML MIG device name containing 'MIG Ng.XXgb'."""
    upper = name.upper()
    marker = upper.find("MIG ")
    haystack = upper[marker + 4 :] if marker >= 0 else upper
    for part in haystack.replace(",", " ").split():
        low = part.lower()
        if "g." not in low:
            continue
        parsed = _profile_gi_slice(low)
        if parsed is not None:
            return parsed
    return None


def infer_profile_from_memory_mb(
    memory_mb: Optional[float],
    gpu_family: Optional[str] = None,
    gi_slice: Optional[int] = None,
) -> tuple:
    """
    Map NVML-reported MIG memory to the nearest known profile.
    Returns (profile, memory_gb, compute_slice).

    When two profiles share the same memory (e.g. 3g.90gb / 4g.90gb),
    `gi_slice` (the Ng in Ng.XXgb) breaks the tie. Without it, memory-
    only matching would always pick the first equal-distance row.
    """
    if memory_mb is None or memory_mb <= 0:
        return ("unknown", 0, 0)
    memory_gb = max(1, round(float(memory_mb) / 1024))
    table = _MIG_PROFILES_BY_FAMILY.get(gpu_family or "A100", _MIG_PROFILES_A100)
    best_dist = min(abs(row[1] - memory_gb) for row in table)
    candidates = [row for row in table if abs(row[1] - memory_gb) == best_dist]
    if len(candidates) == 1:
        return candidates[0]
    if gi_slice is not None:
        matched = [row for row in candidates if _profile_gi_slice(row[0]) == gi_slice]
        if matched:
            return matched[0]
    return candidates[0]


def partition_to_wire(
    entry: Dict[str, object],
    gpu_index: int,
    gpu_family: Optional[str] = None,
) -> Dict[str, object]:
    """Convert a get_mig_partitions() entry to the sync wire shape."""
    uuid = entry.get("uuid")
    idx = entry.get("index", 0)
    partition_id = str(uuid) if uuid else f"mig-{gpu_index}-{idx}"
    gi_raw = entry.get("gi_slice")
    gi_slice = int(gi_raw) if isinstance(gi_raw, (int, float)) else None
    profile, memory_gb, compute_slice = infer_profile_from_memory_mb(
        entry.get("memory_total_mb"),  # type: ignore[arg-type]
        gpu_family=gpu_family,
        gi_slice=gi_slice,
    )
    return {
        "partition_id": partition_id,
        "profile": profile,
        "memory_mb": int(round(float(entry.get("memory_total_mb") or 0))),
        "memory_gb": memory_gb,
        "compute_slice": compute_slice,
        "gpu_index": gpu_index,
    }


def inspect_device(handle, index: int) -> Dict[str, object]:
    """
    Aggregated MIG + process snapshot for a single device.

    Shape:
      {
        gpu_index, gpu_uuid,
        mig_enabled, mig_pending,
        partitions: [...],
        running_processes: [...]
      }
    """
    uuid = None
    try:
        uuid_raw = pynvml.nvmlDeviceGetUUID(handle)
        uuid = uuid_raw.decode() if isinstance(uuid_raw, bytes) else uuid_raw
    except Exception:
        pass

    gpu_name = _device_name(handle)
    gpu_family = gpu_model_key.detect_gpu_family(gpu_name)
    mode = get_mig_mode(handle)
    return {
        'gpu_index': index,
        'gpu_uuid': uuid,
        'gpu_name': gpu_name,
        'gpu_family': gpu_family,
        'mig_supported': device_supports_mig(handle),
        'mig_enabled': mode['current_enabled'],
        'mig_pending': mode['pending_enabled'],
        'partitions': get_mig_partitions(handle) if mode['current_enabled'] else [],
        'running_processes': get_running_processes(handle),
    }
