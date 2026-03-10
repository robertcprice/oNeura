# 25K Pong Near-Real-Time Path

Date: March 10, 2026

## Current measured baseline

Canonical 25K Pong numbers on the current measured runs:

- Mac Rust/Metal, no interval biology: `0.034710x` real time, `28.81x` slower than real time
- NVIDIA A100-PCIE-40GB, batched CUDA path: `0.055905x` real time, `17.89x` slower than real time
- NVIDIA H200, batched CUDA path: `0.088365x` real time, `11.32x` slower than real time

These values are from the shared comparison artifacts in
`results/pong_compare_20260310/`.

The important behavioral distinction is:

- the simulator can still be interacted with in real time at the task/control-loop level
- the neural simulation is still materially slower than full biological brain time at 25K scale

## Why the current assay is slow

The problem is not VRAM pressure. The 25K Pong benchmark is latency-bound:

- the Python/CUDA path drives thousands of tiny `rb.step()` calls from the game loop
- even the Rust/Metal and batched CUDA paths still advance many small dependent steps
- interval biology still runs inside the same hot loop unless explicitly disabled
- motor readout can force repeated host-visible buffer access

So the real-time problem is execution shape, not memory footprint.

## High-value path

The shortest path toward "something like real time" without rewriting the full
stack is:

1. Keep the 25K Pong topology and learning protocol shape.
2. Move the hot control loop into Rust.
3. Add a latency benchmark mode with absolutely no interval biology for this
   assay:
   - circadian
   - pharmacology
   - gene expression
   - metabolism
   - microtubules / Orch-OR
   - glia
4. Avoid per-step motor readback by measuring cumulative spike-count deltas
   across the full stimulus window.

This is intentionally narrower than a full backend rewrite. It improves the
latency path for the CEO-facing Pong comparison while preserving the same basic
task structure.

## What the no-interval-biology runs showed

Removing interval biology helped a little on the Mac native path, but it was
not the main unlock:

- Mac Rust/Metal canonical: `128.085 s`, `0.031307x` real time
- Mac Rust/Metal no interval biology: `115.097 s`, `0.034710x` real time

That is only about a `1.11x` wall-clock speedup.

On CUDA, a stricter no-interval `--benchmark-mode` path was explicitly not the
winner:

- A100 benchmark mode: `127.199 s`, `0.031329x` real time
- H200 benchmark mode: `69.642 s`, `0.057221x` real time

So the current evidence says the main bottleneck is still the core
neuron/synapse execution path and device orchestration, not interval biology by
itself.

## Benchmark-mode caveat

Latency benchmark mode is not the canonical CEO number.

It is a useful engineering comparison for answering:

- how much latency comes from the task loop itself
- how much latency comes from interval biology outside the core Pong assay

But the canonical CEO comparison should continue to use the full measured
baseline unless benchmark mode is explicitly labeled.

## Implemented here

This repo now includes a Rust/Metal Pong fast path in
`oneuro-metal/src/dishbrain_pong.rs` with:

- Rust-owned Pong execution
- latency benchmark mode toggle
- lower-overhead stimulus-window spike counting
- batched CUDA Pong control-loop execution on the Python benchmark path

The CLI entrypoint is:

```bash
cargo run --release --bin dishbrain_pong -- --scale large --rallies 80 --seed 42
```

Latency benchmark mode with no interval biology is:

```bash
cargo run --release --bin dishbrain_pong -- --scale large --rallies 80 --seed 42 --no-interval-biology
```
