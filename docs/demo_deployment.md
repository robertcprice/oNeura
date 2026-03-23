# Demo Deployment

## Recommended Initial Topology

For the first public demo, keep the terrarium simulation authoritative on one
server and split only non-authoritative work:

- `terrarium_web` on one cloud instance/container
- existing `oneura.ai` marketing site unchanged
- `demo.oneura.ai` or `terrarium.oneura.ai` pointing to the terrarium service
- optional background worker later for tournament/evolution/export jobs
- optional Raspberry Pi only as a kiosk/display client, not as a simulation shard

This matches the current codebase better than trying to partition one tightly
coupled frame loop across multiple computers.

## Local Container Test

Build and run:

```bash
docker build -t oneura-terrarium .
docker run --rm -p 8420:8420 \
  -e PORT=8420 \
  -e SEED=42 \
  -e FPS=10 \
  oneura-terrarium
```

Health check:

```bash
curl http://localhost:8420/healthz
```

Demo UI:

```text
http://localhost:8420/
```

## Cloud Shape

Use the cloud service for:

- authoritative simulation tick
- websocket fanout to browsers
- read-only snapshot endpoints

Do not use multiple hosts for one live simulation yet. If you need more
capacity, first scale up the single simulation host and offload background jobs.

## When To Add A Second Machine

Add a separate worker only for tasks that do not need per-frame authority:

- tournament runs
- evolution batches
- export bundle generation
- offline profiling
- capture/render jobs

Keep the live terrarium world authoritative on one machine until the substrate,
packet, explicit-cell, and organism loops are partitionable with explicit halo
exchange.
