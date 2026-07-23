"""Lightweight CPU, RAM, and NVIDIA GPU resource monitoring."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass, field

import psutil


@dataclass
class ResourceMonitor:
    interval_seconds: float = 0.25
    _samples: list[dict[str, float]] = field(default_factory=list)
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self) -> None:
        self._samples = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._collect, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds * 4)
        return summarize_resource_samples(self._samples)

    def _collect(self) -> None:
        process = psutil.Process(os.getpid())
        psutil.cpu_percent(interval=None)
        while not self._stop_event.is_set():
            virtual_memory = psutil.virtual_memory()
            sample = {
                "cpu_percent": psutil.cpu_percent(interval=None),
                "ram_used_mb": virtual_memory.used / (1024 * 1024),
                "ram_percent": virtual_memory.percent,
                "process_rss_mb": process.memory_info().rss / (1024 * 1024),
            }
            sample.update(_read_nvidia_gpu())
            self._samples.append(sample)
            time.sleep(self.interval_seconds)


def _read_nvidia_gpu() -> dict[str, float]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return {"gpu_util_percent": 0.0, "gpu_memory_used_mb": 0.0}

    if result.returncode != 0 or not result.stdout.strip():
        return {"gpu_util_percent": 0.0, "gpu_memory_used_mb": 0.0}

    gpu_utils = []
    gpu_memory = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2:
            gpu_utils.append(float(parts[0]))
            gpu_memory.append(float(parts[1]))

    return {
        "gpu_util_percent": max(gpu_utils) if gpu_utils else 0.0,
        "gpu_memory_used_mb": max(gpu_memory) if gpu_memory else 0.0,
    }


def summarize_resource_samples(samples: list[dict[str, float]]) -> dict[str, float]:
    if not samples:
        return {}

    summary: dict[str, float] = {}
    for key in samples[0]:
        values = [sample[key] for sample in samples if key in sample]
        if values:
            summary[f"avg_{key}"] = sum(values) / len(values)
            summary[f"max_{key}"] = max(values)
    summary["samples"] = len(samples)
    return summary
