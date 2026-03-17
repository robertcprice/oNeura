#!/usr/bin/env python3
"""Compare local Mac and A100 profile/telemetry JSON for CEO-ready summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _load_json(path: str | None) -> Dict[str, Any] | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _nested(data: Dict[str, Any] | None, *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _gpu_summary(data: Dict[str, Any] | None) -> Dict[str, Any] | None:
    summary = _nested(data, "result", "gpu_telemetry", "summary")
    if not summary:
        summary = _nested(data, "gpu_telemetry", "summary")
    if not summary:
        return None
    if isinstance(summary, list) and summary:
        return summary[0]
    return None


def _extract(label: str, profile: Dict[str, Any] | None, telemetry: Dict[str, Any] | None) -> Dict[str, Any]:
    result = _nested(profile, "result") if profile else None
    meta = _nested(profile, "meta") if profile else None
    gpu = _gpu_summary(profile) or _gpu_summary(telemetry)
    utilization = None
    memory_used_mb = None
    memory_util_pct = None
    power_w = None
    if gpu:
        utilization = _nested(gpu, "utilization_gpu_pct", "avg")
        memory_used_mb = _nested(gpu, "memory_used_mb", "max")
        memory_util_pct = _nested(gpu, "memory_utilization_pct", "max")
        power_w = _nested(gpu, "power_draw_w", "avg")
    return {
        "label": label,
        "platform": meta.get("platform") if isinstance(meta, dict) else None,
        "backend": (
            meta.get("backend")
            if isinstance(meta, dict) and meta.get("backend") is not None
            else meta.get("global_backend_hint") if isinstance(meta, dict) else None
        ),
        "measurement_kind": result.get("measurement_kind") if isinstance(result, dict) else None,
        "device_type": result.get("device_type") if isinstance(result, dict) else None,
        "neurons": result.get("neurons") if isinstance(result, dict) else None,
        "synapses": result.get("synapses") if isinstance(result, dict) else None,
        "steps": result.get("steps") if isinstance(result, dict) else None,
        "bio_time_ms": result.get("bio_time_ms") if isinstance(result, dict) else None,
        "wall_time_s": result.get("wall_time_s") if isinstance(result, dict) else None,
        "steps_per_s": result.get("steps_per_s") if isinstance(result, dict) else None,
        "realtime_factor": result.get("realtime_factor") if isinstance(result, dict) else None,
        "process_peak_rss_gb": result.get("process_peak_rss_gb") if isinstance(result, dict) else None,
        "torch_peak_allocated_gb": _nested(result, "torch_memory", "peak_allocated_gb"),
        "torch_peak_reserved_gb": _nested(result, "torch_memory", "peak_reserved_gb"),
        "gpu_util_avg_pct": utilization,
        "gpu_memory_used_max_mb": memory_used_mb,
        "gpu_memory_util_max_pct": memory_util_pct,
        "gpu_power_avg_w": power_w,
        "gpu_context_initialized": result.get("gpu_context_initialized") if isinstance(result, dict) else None,
        "gpu_dispatch_active": result.get("gpu_dispatch_active") if isinstance(result, dict) else None,
        "note": result.get("note") if isinstance(result, dict) else None,
    }


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _text_table(rows: list[Dict[str, Any]]) -> str:
    headers = [
        "run",
        "kind",
        "neurons",
        "synapses",
        "wall_s",
        "bio_ms",
        "rt_factor",
        "steps_s",
        "rss_gb",
        "gpu_util_avg",
        "gpu_mem_max_mb",
    ]
    values = []
    for row in rows:
        values.append(
            [
                row["label"],
                _fmt(row["measurement_kind"]),
                _fmt(row["neurons"], 0),
                _fmt(row["synapses"], 0),
                _fmt(row["wall_time_s"]),
                _fmt(row["bio_time_ms"]),
                _fmt(row["realtime_factor"], 4),
                _fmt(row["steps_per_s"]),
                _fmt(row["process_peak_rss_gb"]),
                _fmt(row["gpu_util_avg_pct"]),
                _fmt(row["gpu_memory_used_max_mb"]),
            ]
        )
    widths = [len(header) for header in headers]
    for row in values:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    lines = []
    lines.append("  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    lines.append("  ".join("-" * widths[idx] for idx in range(len(headers))))
    for row in values:
        lines.append("  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)))
    return "\n".join(lines)


def _comparison_summary(rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    if len(rows) != 2:
        return {}
    left, right = rows

    def _ratio(lhs: Any, rhs: Any) -> float | None:
        if not isinstance(lhs, (int, float)) or not isinstance(rhs, (int, float)) or rhs == 0:
            return None
        return lhs / rhs

    return {
        "measurement_kind_match": left.get("measurement_kind") == right.get("measurement_kind"),
        "realtime_factor_ratio_left_over_right": _ratio(left.get("realtime_factor"), right.get("realtime_factor")),
        "steps_per_s_ratio_left_over_right": _ratio(left.get("steps_per_s"), right.get("steps_per_s")),
        "wall_time_ratio_left_over_right": _ratio(left.get("wall_time_s"), right.get("wall_time_s")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Mac and A100 oNeura profile JSON")
    parser.add_argument("--mac-profile", required=True, help="Mac profile JSON path")
    parser.add_argument("--mac-telemetry", default=None, help="Optional Mac telemetry JSON path")
    parser.add_argument("--a100-profile", default=None, help="A100 profile JSON path")
    parser.add_argument("--a100-telemetry", default=None, help="Optional A100 telemetry JSON path")
    parser.add_argument("--json", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    mac = _extract("mac", _load_json(args.mac_profile), _load_json(args.mac_telemetry))
    rows = [mac]
    if args.a100_profile or args.a100_telemetry:
        rows.append(_extract("a100", _load_json(args.a100_profile), _load_json(args.a100_telemetry)))

    payload = {
        "runs": rows,
        "table": _text_table(rows),
        "comparison": _comparison_summary(rows),
    }
    print(payload["table"])
    print()
    print(json.dumps(payload, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
