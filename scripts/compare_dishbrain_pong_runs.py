#!/usr/bin/env python3
"""Compare analyzed DishBrain Pong runs and optional GPU/process telemetry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: str | None) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _gpu_summary(telemetry: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not telemetry:
        return None
    payload = telemetry.get("gpu_telemetry", telemetry)
    summary = payload.get("summary")
    if isinstance(summary, list) and summary:
        return summary[0]
    return None


def _to_gb_from_bytes(value: int | float | None) -> Optional[float]:
    if value is None:
        return None
    return float(value) / (1024.0 ** 3)


def _row(
    label: str,
    analysis: Dict[str, Any],
    telemetry: Optional[Dict[str, Any]],
    peak_memory_bytes: int | None,
    note: str | None,
) -> Dict[str, Any]:
    gpu = _gpu_summary(telemetry)
    slower = analysis.get("slower_than_real_time")
    if slower is None and analysis.get("realtime_factor") not in (None, 0):
        slower = 1.0 / float(analysis["realtime_factor"])

    row = {
        "label": label,
        "scale": analysis.get("scale"),
        "requested_device": analysis.get("requested_device"),
        "backend": analysis.get("backend"),
        "gpu": analysis.get("gpu"),
        "wall_time_s": analysis.get("wall_time_s"),
        "bio_time_ms": analysis.get("bio_time_ms"),
        "realtime_factor": analysis.get("realtime_factor"),
        "slower_than_real_time": slower,
        "neural_steps_per_s": analysis.get("neural_steps_per_s"),
        "rallies": analysis.get("rallies"),
        "hits": analysis.get("hits"),
        "misses": analysis.get("misses"),
        "process_peak_memory_gb": _to_gb_from_bytes(peak_memory_bytes),
        "note": note,
    }
    if gpu:
        mem_min = gpu["memory_used_mb"]["min"]
        mem_max = gpu["memory_used_mb"]["max"]
        row.update(
            {
                "gpu_util_avg_pct": gpu["utilization_gpu_pct"]["avg"],
                "gpu_util_max_pct": gpu["utilization_gpu_pct"]["max"],
                "gpu_mem_used_max_mb": mem_max,
                "gpu_mem_used_delta_mb": None
                if mem_min is None or mem_max is None
                else mem_max - mem_min,
                "gpu_power_avg_w": gpu["power_draw_w"]["avg"],
                "gpu_temp_avg_c": gpu["temperature_c"]["avg"],
                "gpu_samples": gpu["sample_count"],
            }
        )
    return row


def _table(rows: List[Dict[str, Any]]) -> str:
    headers = [
        "run",
        "gpu",
        "wall_s",
        "bio_ms",
        "rt_factor",
        "slowdown",
        "steps_s",
        "gpu_util_avg",
        "vram_max_mb",
        "vram_delta_mb",
        "proc_peak_gb",
    ]
    values: List[List[str]] = []
    for row in rows:
        values.append(
            [
                row["label"],
                _fmt(row.get("gpu")),
                _fmt(row.get("wall_time_s")),
                _fmt(row.get("bio_time_ms"), 1),
                _fmt(row.get("realtime_factor"), 6),
                _fmt(row.get("slower_than_real_time"), 1),
                _fmt(row.get("neural_steps_per_s"), 1),
                _fmt(row.get("gpu_util_avg_pct"), 1),
                _fmt(row.get("gpu_mem_used_max_mb"), 0),
                _fmt(row.get("gpu_mem_used_delta_mb"), 0),
                _fmt(row.get("process_peak_memory_gb"), 2),
            ]
        )
    widths = [len(h) for h in headers]
    for row in values:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    lines = []
    lines.append("  ".join(h.ljust(widths[idx]) for idx, h in enumerate(headers)))
    lines.append("  ".join("-" * widths[idx] for idx in range(len(headers))))
    for row in values:
        lines.append("  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)))
    return "\n".join(lines)


def _find_row(rows: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    for row in rows:
        if row["label"] == label:
            return row
    raise ValueError(f"Unknown label: {label}")


def _ratio(numerator: Any, denominator: Any) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _comparison(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = {row["label"] for row in rows}
    out: Dict[str, Any] = {}
    if {"h200", "a100"} <= labels:
        h200 = _find_row(rows, "h200")
        a100 = _find_row(rows, "a100")
        out["h200_vs_a100"] = {
            "wall_speedup": _ratio(a100["wall_time_s"], h200["wall_time_s"]),
            "realtime_factor_ratio": _ratio(h200["realtime_factor"], a100["realtime_factor"]),
        }
    if {"a100", "mac"} <= labels:
        a100 = _find_row(rows, "a100")
        mac = _find_row(rows, "mac")
        out["a100_vs_mac"] = {
            "wall_speedup": _ratio(mac["wall_time_s"], a100["wall_time_s"]),
            "realtime_factor_ratio": _ratio(a100["realtime_factor"], mac["realtime_factor"]),
        }
    if {"h200", "mac"} <= labels:
        h200 = _find_row(rows, "h200")
        mac = _find_row(rows, "mac")
        out["h200_vs_mac"] = {
            "wall_speedup": _ratio(mac["wall_time_s"], h200["wall_time_s"]),
            "realtime_factor_ratio": _ratio(h200["realtime_factor"], mac["realtime_factor"]),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mac-analysis", required=True)
    parser.add_argument("--mac-note", default="2-rally smoke run on patched MPS path")
    parser.add_argument("--mac-peak-memory-bytes", type=int, default=None)
    parser.add_argument("--a100-analysis", required=True)
    parser.add_argument("--a100-telemetry", required=True)
    parser.add_argument("--h200-analysis", required=True)
    parser.add_argument("--h200-telemetry", required=True)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    rows = [
        _row(
            "mac",
            _load_json(args.mac_analysis) or {},
            None,
            args.mac_peak_memory_bytes,
            args.mac_note,
        ),
        _row(
            "a100",
            _load_json(args.a100_analysis) or {},
            _load_json(args.a100_telemetry),
            None,
            "full 80-rally run with sampled nvidia-smi telemetry",
        ),
        _row(
            "h200",
            _load_json(args.h200_analysis) or {},
            _load_json(args.h200_telemetry),
            None,
            "full 80-rally run with sampled nvidia-smi telemetry",
        ),
    ]

    payload = {
        "runs": rows,
        "table": _table(rows),
        "comparison": _comparison(rows),
    }
    print(payload["table"])
    print()
    print(json.dumps(payload, indent=2))

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
