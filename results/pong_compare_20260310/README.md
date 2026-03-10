## 25K Pong Comparison

Date: March 10, 2026

Workload:
- DishBrain Pong replication, Experiment 1
- `"large"` scale brain
- About `25K` neurons / about `7.0M` synapses

### Current best measured runs

| System | Scope | Wall time | Biological time | Real-time factor | Slowdown vs real time | Resource data |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| MacBook (Rust/Metal, no interval biology) | Full 80-rally run | 115.097 s | 3995.0 ms | 0.034710x | 28.81x slower | Peak process memory footprint 1.24 GiB |
| NVIDIA A100-PCIE-40GB (batched CUDA path) | Full 80-rally run | 71.819 s | 4015.0 ms | 0.055905x | 17.89x slower | GPU util avg 54.6%, GPU util max 93.0%, VRAM max 2.15 GB, power avg 136.2 W |
| NVIDIA H200 (batched CUDA path) | Full 80-rally run | 45.210 s | 3995.0 ms | 0.088365x | 11.32x slower | GPU util avg 41.1%, GPU util max 93.0%, VRAM max 2.14 GB, power avg 230.1 W |

### Direct takeaways

- The updated batched CUDA Pong path materially improved the 25K GPU runs without changing the experiment semantics.
- Relative to the older measured CUDA runs, the batched path improved the A100 by about `1.28x` in both wall time and real-time factor and improved the H200 by about `1.62x`.
- On the current best measured runs, the A100 is about `1.61x` closer to real time than the best current Mac local run, and the H200 is about `2.55x` closer to real time than the Mac.
- The H200 is about `1.58x` faster than the A100 on this workload.
- Even the fastest current result is still not real time. The H200 best run is `0.088365x` real time, which is still about `11.3x` slower than biological real time.
- The benchmark can still be run interactively as a live closed-loop task, but the neural simulation underneath is slower than full biological brain speed on all measured systems.
- VRAM remains low because this workload is small relative to these GPUs. The bottleneck is execution shape and sequential timestep orchestration, not card capacity.

### No-interval benchmark-mode check

- We also reran a CUDA `--benchmark-mode` path that disables interval biology and uses the reduced-overhead fast path.
- On current CUDA, that mode was **not** the speed winner:
  - A100 benchmark mode: `127.199 s`, `0.031329x` real time, GPU util avg `70.6%`, VRAM max `1.27 GB`
  - H200 benchmark mode: `69.642 s`, `0.057221x` real time, GPU util avg `67.1%`, VRAM max `1.39 GB`
- The practical conclusion is that removing interval biology alone is not the main unlock on the current CUDA backend. The bigger win came from batching the Pong hot loop while staying on the normal compiled CUDA path.

### Source artifacts

- Mac Rust/Metal no-interval run: [dishbrain_large_metal_rust_no_interval.json](/Users/bobbyprice/projects/oNeuro/results/pong_compare_20260310/dishbrain_large_metal_rust_no_interval.json)
- Mac Rust/Metal no-interval analysis: [dishbrain_large_metal_rust_no_interval_analysis.json](/Users/bobbyprice/projects/oNeuro/results/pong_compare_20260310/dishbrain_large_metal_rust_no_interval_analysis.json)
- Mac Rust/Metal no-interval log: [dishbrain_large_metal_rust_no_interval.txt](/Users/bobbyprice/projects/oNeuro/results/pong_compare_20260310/dishbrain_large_metal_rust_no_interval.txt)
- A100 batched run: [dishbrain_large_a100_batched.json](/Users/bobbyprice/projects/oNeuro/results/pong_compare_20260310/dishbrain_large_a100_batched.json)
- A100 batched analysis: [dishbrain_large_a100_batched_analysis.json](/Users/bobbyprice/projects/oNeuro/results/pong_compare_20260310/dishbrain_large_a100_batched_analysis.json)
- A100 batched telemetry: [dishbrain_large_a100_batched_gpu.json](/Users/bobbyprice/projects/oNeuro/results/pong_compare_20260310/dishbrain_large_a100_batched_gpu.json)
- A100 batched log: [dishbrain_large_a100_batched.txt](/Users/bobbyprice/projects/oNeuro/results/pong_compare_20260310/dishbrain_large_a100_batched.txt)
- H200 batched run: [dishbrain_large_h200_batched.json](/Users/bobbyprice/projects/oNeuro/results/pong_compare_20260310/dishbrain_large_h200_batched.json)
- H200 batched analysis: [dishbrain_large_h200_batched_analysis.json](/Users/bobbyprice/projects/oNeuro/results/pong_compare_20260310/dishbrain_large_h200_batched_analysis.json)
- H200 batched telemetry: [dishbrain_large_h200_batched_gpu.json](/Users/bobbyprice/projects/oNeuro/results/pong_compare_20260310/dishbrain_large_h200_batched_gpu.json)
- H200 batched log: [dishbrain_large_h200_batched.txt](/Users/bobbyprice/projects/oNeuro/results/pong_compare_20260310/dishbrain_large_h200_batched.txt)
- A100 benchmark-mode run: [dishbrain_large_a100_benchmark.json](/Users/bobbyprice/projects/oNeuro/results/pong_compare_20260310/dishbrain_large_a100_benchmark.json)
- A100 benchmark-mode analysis: [dishbrain_large_a100_benchmark_analysis.json](/Users/bobbyprice/projects/oNeuro/results/pong_compare_20260310/dishbrain_large_a100_benchmark_analysis.json)
- A100 benchmark-mode telemetry: [dishbrain_large_a100_benchmark_gpu.json](/Users/bobbyprice/projects/oNeuro/results/pong_compare_20260310/dishbrain_large_a100_benchmark_gpu.json)
- H200 benchmark-mode run: [dishbrain_large_h200_benchmark.json](/Users/bobbyprice/projects/oNeuro/results/pong_compare_20260310/dishbrain_large_h200_benchmark.json)
- H200 benchmark-mode analysis: [dishbrain_large_h200_benchmark_analysis.json](/Users/bobbyprice/projects/oNeuro/results/pong_compare_20260310/dishbrain_large_h200_benchmark_analysis.json)
- H200 benchmark-mode telemetry: [dishbrain_large_h200_benchmark_gpu.json](/Users/bobbyprice/projects/oNeuro/results/pong_compare_20260310/dishbrain_large_h200_benchmark_gpu.json)
- Batched comparison summary: [dishbrain_pong_batched_compare_summary.json](/Users/bobbyprice/projects/oNeuro/results/pong_compare_20260310/dishbrain_pong_batched_compare_summary.json)
- Benchmark-mode comparison summary: [dishbrain_pong_benchmark_mode_compare_summary.json](/Users/bobbyprice/projects/oNeuro/results/pong_compare_20260310/dishbrain_pong_benchmark_mode_compare_summary.json)
- Main comparison summary: [dishbrain_pong_compare_summary.json](/Users/bobbyprice/projects/oNeuro/results/pong_compare_20260310/dishbrain_pong_compare_summary.json)
