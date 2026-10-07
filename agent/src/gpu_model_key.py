"""
Canonical gpu_type → model_key resolution aligned with gpu-service
catalog/gpu-model-key.ts.

The agent reports gpu_type strings; gpu-service resolves them to catalog
keys (e.g. h200-sxm, b200-sxm).
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

# Mirrors gpu.gpu_model_catalog seed (migrations 015 + 037).
HARDWARE_CATALOG_KEYS = frozenset(
    {
        "h200-sxm",
        "h100-sxm",
        "h100-pcie",
        "a100-80gb",
        "a100-40gb",
        "l40s",
        "t4",
        "amd-mi300x",
        "b200-sxm",
        "gb200",
    }
)

GpuFamily = Literal["A100", "H100", "H200", "B200", "GB200"]
ModelKeyMiss = Literal["no_gpu_type", "ambiguous_sku", "unknown_gpu_type"]


def detect_gpu_family(gpu_type: str | None) -> Optional[GpuFamily]:
    """Map a raw gpu_type / NVML name to a short GPU family."""
    if not gpu_type or not gpu_type.strip():
        return None
    norm = gpu_type.strip().upper()
    # Longest-family-first — GB200 before B200, H200 before H100, L40S before L4.
    # Do not match bare "BLACKWELL" — that also hits workstation SKUs such as
    # "RTX PRO 6000 Blackwell", which must stay unpriced (no catalog row).
    if "GB200" in norm or "GRACE BLACKWELL" in norm:
        return "GB200"
    if "B200" in norm:
        return "B200"
    if "H200" in norm:
        return "H200"
    if "H100" in norm:
        return "H100"
    if "A100" in norm:
        return "A100"
    return None


def normalize_gpu_type(gpu_type: str | None) -> str | None:
    """Return a stable marketing gpu_type for sync (matches AWS monitor labels)."""
    family = detect_gpu_family(gpu_type)
    if family:
        return family
    if gpu_type and gpu_type.strip():
        return gpu_type.strip()
    return None


def is_mig_capable(gpu_type: str | None) -> bool:
    """Hopper and Blackwell (B200/GB200 GPU) support MIG."""
    family = detect_gpu_family(gpu_type)
    return family in ("A100", "H100", "H200", "B200", "GB200")


def resolve_gpu_model_key(
    gpu_type: str | None,
) -> Tuple[Optional[str], Optional[ModelKeyMiss]]:
    """
    Resolve gpu_type to a gpu.gpu_model_catalog model_key.

    Returns (key, miss_reason). miss_reason is None on hit.
    """
    if not gpu_type or not gpu_type.strip():
        return None, "no_gpu_type"

    raw = gpu_type.strip()
    norm = raw.upper()

    if "MI300X" in norm:
        return "amd-mi300x", None

    family = detect_gpu_family(raw)
    if family == "GB200":
        return "gb200", None
    if family == "B200":
        return "b200-sxm", None

    if family == "H200":
        return "h200-sxm", None

    if "H100" in norm:
        if _is_pcie(norm):
            return "h100-pcie", None
        if _is_sxm(norm):
            return "h100-sxm", None
        return None, "ambiguous_sku"

    if "A100" in norm:
        if _a100_is_80gb(norm):
            return "a100-80gb", None
        if _a100_is_40gb(norm):
            return "a100-40gb", None
        return None, "ambiguous_sku"

    if "L40S" in norm:
        return "l40s", None
    if _is_l4(norm):
        return "l4", None
    if _is_t4(norm):
        return "t4", None

    return None, "unknown_gpu_type"


def _is_pcie(norm: str) -> bool:
    return "PCIE" in norm or "PCI-E" in norm


def _is_sxm(norm: str) -> bool:
    if "NVL" in norm:
        return False
    return "SXM" in norm or "HGX" in norm or "HBM3" in norm


def _a100_is_80gb(norm: str) -> bool:
    return "80GB" in norm or "-80GB" in norm or "80 GB" in norm


def _a100_is_40gb(norm: str) -> bool:
    return "40GB" in norm or "-40GB" in norm or "40 GB" in norm


def _is_l4(norm: str) -> bool:
    import re

    return bool(re.search(r"\bL4\b", norm) or re.search(r"L4(?![0-9S])", norm))


def _is_t4(norm: str) -> bool:
    import re

    return bool(re.search(r"\bT4\b", norm) or re.search(r"T4(?![0-9])", norm))
