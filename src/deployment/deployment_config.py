"""Target hardware profiles and optimization settings."""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class HardwareProfile:
    name: str
    vram_budget_gb: float
    recommended_precision: str
    max_engine_workspace_gb: float
    target_latency_ms: float
    description: str

PROFILES = {
    "rtx_5050_laptop": HardwareProfile(
        name="rtx_5050_laptop",
        vram_budget_gb=8.0,
        recommended_precision="fp16", # 5th-gen Tensor cores
        max_engine_workspace_gb=2.0, # Do not starve OS/dashboard
        target_latency_ms=50.0,
        description="Primary Target: NVIDIA RTX 5050 Laptop (Blackwell GB207). 8GB VRAM hard limit."
    ),
    "jetson_orin_nano": HardwareProfile(
        name="jetson_orin_nano",
        vram_budget_gb=8.0, # System RAM is shared
        recommended_precision="fp16",
        max_engine_workspace_gb=2.0,
        target_latency_ms=100.0,
        description="NVIDIA Jetson Orin Nano"
    ),
    "jetson_orin_nx": HardwareProfile(
        name="jetson_orin_nx",
        vram_budget_gb=16.0,
        recommended_precision="fp16",
        max_engine_workspace_gb=4.0,
        target_latency_ms=100.0,
        description="NVIDIA Jetson Orin NX"
    ),
    "jetson_agx_orin": HardwareProfile(
        name="jetson_agx_orin",
        vram_budget_gb=32.0,
        recommended_precision="int8",
        max_engine_workspace_gb=8.0,
        target_latency_ms=50.0,
        description="NVIDIA Jetson AGX Orin"
    ),
    "desktop_gpu": HardwareProfile(
        name="desktop_gpu",
        vram_budget_gb=24.0,
        recommended_precision="fp16",
        max_engine_workspace_gb=8.0,
        target_latency_ms=30.0,
        description="Generic high-end desktop GPU (e.g. RTX 4090)"
    )
}

def get_profile(name: str) -> HardwareProfile:
    if name not in PROFILES:
        raise ValueError(f"Unknown hardware profile '{name}'. Valid profiles: {list(PROFILES.keys())}")
    return PROFILES[name]
