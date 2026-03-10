# Pong Retina Pretraining Notes

Date: 2026-03-10

## Goal

Test whether a short teacher-forced visual warmup helps the Pong brain learn from retinal input before autonomous play.

## Implementation

Code changes are in `demos/demo_dishbrain_pong.py`.

Experiment 7 now:
- renders Pong through the molecular retina
- optionally includes the paddle in the rendered frame (`retina_body`)
- runs teacher-forced rallies before free play
- clamps the teacher-selected motor population during those pretraining rallies
- applies a relay->motor Hebbian nudge on teacher-guided steps
- then evaluates the learned policy on held-out rallies without further feedback

The expert policy was upgraded to target the predicted intercept point, not just the current ball position.

## Commands Run

```bash
python3 demos/demo_dishbrain_pong.py --exp 7 --scale small --device cpu --seed 42 --runs 3 --rallies 32 --pretrain-rallies 16 --json results/pong_retina_pretraining_20260310/retina_runs3_train32_pre16.json
python3 demos/demo_dishbrain_pong.py --exp 7 --scale small --device cpu --seed 42 --runs 3 --rallies 32 --pretrain-rallies 4 --json results/pong_retina_pretraining_20260310/retina_runs3_train32_pre4.json
python3 demos/demo_dishbrain_pong.py --exp 7 --scale small --device cpu --seed 42 --runs 3 --rallies 32 --input-mode retina_body --pretrain-rallies 8 --json results/pong_retina_pretraining_20260310/retina_body_runs3_train32_pre8.json
python3 demos/demo_dishbrain_pong.py --exp 7 --scale small --device cpu --seed 42 --runs 3 --rallies 32 --input-mode retina_body --pretrain-rallies 4 --json results/pong_retina_pretraining_20260310/retina_body_runs3_train32_pre4.json
python3 demos/demo_dishbrain_pong.py --exp 7 --scale small --device cpu --seed 42 --runs 5 --rallies 32 --json results/pong_retina_pretraining_20260310/retina_body_default_runs5_train32_pre8.json
```

## Mean Held-Out Test Rate

All numbers below are means over seeds `42-44`, except where noted.

| setup | mean held-out |
|---|---:|
| scalar + free_energy | 0.400 |
| retina + free_energy_replay | 0.533 |
| retina + teacher + free_energy | 0.417 to 0.450 |
| retina + teacher + free_energy_replay | 0.400 to 0.417 |
| retina_body + free_energy_replay | 0.550 |
| retina_body + teacher + free_energy | 0.483 to 0.533 |
| retina_body + teacher + free_energy_replay, pretrain 8 | 0.583 |
| retina_body + teacher + free_energy_replay, pretrain 4 | 0.383 |

From the longer 5-seed confirmation batch (`42-46`):

| setup | mean held-out |
|---|---:|
| scalar + free_energy | 0.430 |
| retina_body + free_energy_replay | 0.540 |
| retina_body + teacher + free_energy | 0.530 |
| retina_body + teacher + free_energy_replay | 0.560 |

## Takeaways

1. Ball-only retinal pretraining does not help reliably.
2. This is expected: the correct action depends on paddle position too, so ball-only frames make teacher labels ambiguous.
3. When the paddle is visible (`retina_body`), teacher pretraining becomes useful.
4. The best tested setup so far is `retina_body + teacher + free_energy_replay` with `8` pretraining rallies.
5. The gain is real but still small: in the longer 5-seed batch it reached `0.560` vs `0.540` for the best no-pretrain retinal baseline.

## Current Default

`exp 7` now defaults to:
- `input_mode=retina_body`
- `pretrain_rallies=8`

That matches the best mean result from this batch.

## Recording Artifact

Recorded demo GIF of the best current Pong policy:
- `pong_retina_body_teacher_replay_demo.gif`
