#!/usr/bin/env python3
"""Helpers for capturing NVIDIA GPU telemetry via nvidia-smi."""

from __future__ import annotations

import csv
import io
import shutil
import statistics
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional

QUERY_FIELDS = (
    "index",
    "name",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
    "temperature.gpu",
)


def _to_float(raw: str) -> Optional[float]:
    value = raw.strip()
    if not value or value in {"N/A", "[Not Supported]"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _stats(values: List[Optional[float]]) -> Dict[str, Optional[float]]:
    numbers = [value for value in values if value is not None]
    if not numbers:
        return {"avg": None, "min": None, "max": None}
    return {
        "avg": statistics.fmean(numbers),
        "min": min(numbers),
        "max": max(numbers),
    }


def query_nvidia_smi() -> Dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {
            "available": False,
            "reason": "nvidia-smi not found",
            "records": [],
        }

    cmd = [
        executable,
        f"--query-gpu={','.join(QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return {
            "available": False,
            "reason": str(exc),
            "records": [],
        }

    timestamp = time.time()
    records: List[Dict[str, Any]] = []
    reader = csv.reader(io.StringIO(proc.stdout))
    for row in reader:
        if not row:
            continue
        index, name, util_gpu, util_mem, mem_used, mem_total, power_draw, temp_gpu = row
        memory_util = None
        mem_used_value = _to_float(mem_used)
        mem_total_value = _to_float(mem_total)
        if mem_used_value is not None and mem_total_value not in (None, 0.0):
            memory_util = (mem_used_value / mem_total_value) * 100.0
        records.append(
            {
                "timestamp": timestamp,
                "index": int(index.strip()),
                "name": name.strip(),
                "utilization_gpu_pct": _to_float(util_gpu),
                "utilization_memory_pct": _to_float(util_mem),
                "memory_used_mb": mem_used_value,
                "memory_total_mb": mem_total_value,
                "memory_utilization_pct": memory_util,
                "power_draw_w": _to_float(power_draw),
                "temperature_c": _to_float(temp_gpu),
            }
        )

    return {"available": True, "reason": None, "records": records}


def summarize_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_gpu: Dict[int, List[Dict[str, Any]]] = {}
    for record in records:
        by_gpu.setdefault(record["index"], []).append(record)

    summary: List[Dict[str, Any]] = []
    for gpu_index in sorted(by_gpu):
        gpu_records = by_gpu[gpu_index]
        summary.append(
            {
                "index": gpu_index,
                "name": gpu_records[0]["name"],
                "sample_count": len(gpu_records),
                "utilization_gpu_pct": _stats([r["utilization_gpu_pct"] for r in gpu_records]),
                "utilization_memory_pct": _stats([r["utilization_memory_pct"] for r in gpu_records]),
                "memory_used_mb": _stats([r["memory_used_mb"] for r in gpu_records]),
                "memory_utilization_pct": _stats([r["memory_utilization_pct"] for r in gpu_records]),
                "power_draw_w": _stats([r["power_draw_w"] for r in gpu_records]),
                "temperature_c": _stats([r["temperature_c"] for r in gpu_records]),
            }
        )
    return summary


class NvidiaSmiSampler:
    """Poll nvidia-smi on a background thread while a workload runs."""

    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self.records: List[Dict[str, Any]] = []
        self.errors: List[str] = []
        self.available = shutil.which("nvidia-smi") is not None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self.available:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_s * 2.0))
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self.interval_s)

    def sample_once(self) -> None:
        result = query_nvidia_smi()
        if not result["available"]:
            if result["reason"]:
                self.errors.append(result["reason"])
            return
        self.records.extend(result["records"])

    def report(self, include_samples: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "available": self.available,
            "interval_s": self.interval_s,
            "sample_count": len(self.records),
            "errors": self.errors,
            "summary": summarize_records(self.records),
        }
        if include_samples:
            payload["samples"] = self.records
        return payload
