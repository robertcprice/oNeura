# Doom Teacher Pretraining Notes

Date: 2026-03-10

## Goal

Port the validated Pong idea into Doom:
- teacher-forced visual warmup before free play
- optional relay replay on positive feedback
- record a real gameplay artifact
- test whether the same mechanism helps enemy-killing, not just health gathering

## Implementation

Code changes are in `demos/demo_doom_vizdoom.py`.

The port adds:
- visible label access from ViZDoom
- a simple `health_gathering` teacher policy that turns toward visible medikits
- relay activation replay inside `DoomFEPProtocol`
- a teacher-forced pretraining episode helper
- active-relay credit assignment so feedback strengthens the current visual pattern, not the whole relay bank
- optional combat-only online teacher shaping during free play
- repo-local video output under `results/doom_videos/`

## Commands Run

```bash
python3 demos/demo_doom_vizdoom.py --exp 1 --scale small --device cpu --scenario health_gathering --episodes 4 --pretrain-episodes 2 --structured-replay-scale 0.8 --video --json results/doom_teacher_pretraining_20260310/teacher_replay_health_gathering.json
python3 demos/demo_doom_vizdoom.py --exp 1 --scale small --device cpu --scenario health_gathering --episodes 4 --json results/doom_teacher_pretraining_20260310/baseline_health_gathering.json
python3 demos/demo_doom_vizdoom.py --exp 1 --scale small --device cpu --scenario health_gathering --episodes 4 --pretrain-episodes 2 --teacher-motor-intensity 60 --teacher-hebbian-delta 1.2 --video --json results/doom_teacher_pretraining_20260310/teacher_tuned_health_gathering.json
```

Multi-seed comparisons were run through the Python API and saved to:
- `multiseed_compare.json`
- `multiseed_compare_tuned.json`
- `teacher_param_sweep_3seed.json`
- `kill_compare_defend_center.json`
- `kill_compare_deadly_corridor_3seed.json`
- `combat_shaping_sweep.json`
- `combat_shaping_compare_defend_5seed.json`
- `kill_record_active_relay_seed42.json`
- `deadly_corridor_teacher_transfer_sweep.json`
- `deadly_corridor_pretrain4_compare_5seed.json`
- `deadly_corridor_pretrain4_seed45.json`
- `deadly_corridor_finishing_sweep.json`
- `deadly_corridor_decoder_bias_sweep.json`
- `deadly_corridor_decoder_penalty_sweep.json`
- `deadly_corridor_decoder_penalty_compare_5seed.json`
- `deadly_corridor_decoder_bias_seed46.json`

## Results

Single recorded seed (`42`):
- baseline total health gained over 4 episodes: `8`
- teacher + replay total health gained over 4 episodes: `16`
- teacher warmup match rate: `32%`

5-seed comparison (`42-46`, 4 episodes each):
- baseline mean total health gained: `6.4`
- teacher + replay mean total health gained: `4.8`

Parameter sweep result:
- best 3-seed config was `teacher_motor_intensity=60`, `teacher_hebbian_delta=1.2`, `structured_replay_scale=0.0`

Tuned single recorded seed (`42`):
- tuned teacher total health gained over 4 episodes: `16`
- tuned teacher warmup match rate: `47%`

Tuned 5-seed comparison (`42-46`, 4 episodes each):
- baseline mean total health gained: `6.4`
- tuned teacher-pretrain mean total health gained: `8.0`
- tuned teacher mean warmup match rate: `35.5%`

Combat comparison: `defend_the_center` (`42-46`, 6 episodes each, primary metric `kills`)
- baseline mean total kills: `2.6`
- tuned teacher-pretrain mean total kills: `2.0`
- tuned teacher mean warmup match rate: `40.6%`
- tuned setup beat baseline on only `2/5` seeds

Combat comparison: `deadly_corridor` (`42-44`, 4 episodes each, primary metric `kills`)
- baseline mean total kills: `1.33`
- tuned teacher-pretrain mean total kills: `0.67`
- tuned teacher mean warmup match rate: `46.7%`
- tuned setup beat baseline on `0/3` seeds

Combat update after active-relay credit assignment: `defend_the_center` (`42-46`, 6 episodes each, primary metric `kills`)
- previous baseline mean total kills: `2.6`
- new active-relay mean total kills: `3.8`
- new active-relay last-quarter mean kills: `0.8`
- new active-relay mean free-play teacher match rate: `16.9%`

Combat shaping sweep (`42-44`) on top of the active-relay fix:
- `defend_the_center`: best total kills stayed at `combat_teacher_shaping_delta=0.0`
- `defend_the_center`: `combat_teacher_shaping_delta=0.3` improved some late kills in the small sweep, but lost on the 5-seed follow-up
- `deadly_corridor`: online combat shaping did not help; all shaped variants were at or below the active-relay baseline

5-seed `defend_the_center` follow-up (`42-46`):
- `combat_teacher_shaping_delta=0.0`: mean total kills `3.8`, mean last-quarter kills `0.8`
- `combat_teacher_shaping_delta=0.3`: mean total kills `3.2`, mean last-quarter kills `0.2`

Corridor teacher-only sanity check (`42-44`, 1 episode each):
- teacher policy kills: `1, 1, 1`
- teacher survived on `2/3` seeds

`deadly_corridor` transfer sweep with the new stateful corridor teacher (`42-44`, 4 episodes each, primary metric `kills`)
- `pretrain_episodes=0`: mean total kills `0.67`, mean survival `0.25`, mean damage `98.0`
- `pretrain_episodes=2`: mean total kills `0.0`, mean survival `0.17`, mean damage `107.5`
- `pretrain_episodes=4`: mean total kills `1.0`, mean survival `0.42`, mean damage `96.0`

5-seed `deadly_corridor` follow-up (`42-46`, 4 episodes each)
- `pretrain_episodes=0`: mean total kills `1.0`, mean survival `0.40`, mean damage `101.4`
- `pretrain_episodes=4`: mean total kills `0.8`, mean survival `0.50`, mean damage `93.3`
- interpretation: the new corridor teacher improves survival and damage, but not kill rate, so it is helping cover/movement transfer more than combat finishing

Finishing-shot sweep (`42-44`, `pretrain_episodes=4`) with attack-window transfer enabled during free play
- base (`combat_attack_window_delta=0.0`, `combat_attack_miss_delta=0.0`): mean total kills `1.0`, mean survival `0.50`, mean damage `90.0`
- `combat_attack_window_delta=0.25`: mean total kills `0.67`, mean survival `0.25`, mean damage `100.5`
- `combat_attack_window_delta=0.5`: mean total kills `0.67`, mean survival `0.33`, mean damage `98.0`
- `combat_attack_window_delta=0.25`, `combat_attack_miss_delta=0.05`: mean total kills `0.33`, mean survival `0.33`, mean damage `99.0`
- `combat_attack_window_delta=0.5`, `combat_attack_miss_delta=0.1`: mean total kills `0.67`, mean survival `0.25`, mean damage `102.0`
- attack-window fire rate stayed low and blind attacks stayed high, so the extra transfer did not solve the finishing problem

Decoder-bias sweeps on top of the corridor teacher:
- first sweep (`42-44`) showed that a decoder prior can materially increase attack-window firing; `bonus=1.0`, `penalty=0.5` raised mean attack-window fire rate from `0.010` to `0.322` and mean kills from `0.33` to `1.33`, but that did not hold on the larger batch
- stronger penalty sweep (`42-44`) found the more balanced setting at `combat_decoder_attack_bonus=1.0`, `combat_decoder_attack_penalty=1.5`

5-seed decoder-bias compare (`42-46`, `pretrain_episodes=4`)
- base: mean total kills `0.8`, mean survival `0.30`, mean damage `96.9`, mean attack-window fire rate `0.042`
- `combat_decoder_attack_bonus=1.0`, `combat_decoder_attack_penalty=1.5`: mean total kills `0.8`, mean survival `0.45`, mean damage `96.0`, mean attack-window fire rate `0.125`
- interpretation: the decoder gate did not raise mean kills on the 5-seed batch, but it did produce a better balanced corridor policy with equal kills, better survival, slightly lower damage, and much more firing in valid attack windows

Per-seed totals from `multiseed_compare.json`:

| seed | baseline | teacher + replay |
|---|---:|---:|
| 42 | 8 | 16 |
| 43 | 16 | 0 |
| 44 | 0 | 0 |
| 45 | 0 | 0 |
| 46 | 8 | 8 |

## Takeaways

1. The Doom port works mechanically: pretraining, replay, JSON output, and recording all run.
2. The teacher policy itself is not the problem. Run directly, it can collect health reliably in `health_gathering`.
3. The transfer bottleneck was pretraining strength. Stronger motor clamping improved teacher/network agreement and mean health.
4. Replay was not helpful in the stronger-transfer regime for this task.
5. The best tested Doom setup so far is `pretrain_episodes=2`, `teacher_motor_intensity=60`, `teacher_hebbian_delta=1.2`, `structured_replay_scale=0.0`.
6. That tuned setup beat the baseline on the current 5-seed batch: `8.0` vs `6.4` mean total health.
7. That same tuned setup does not transfer cleanly to combat. It can get kills, but it did not beat the baseline on `defend_the_center` or `deadly_corridor`.
8. So the current biological pretraining helps navigation/collection more than combat; enemy-killing still needs a better transfer mechanism or a better combat-specific teacher.
9. The relay-targeted credit fix does help combat in `defend_the_center`: mean kills improved from `2.6` to `3.8` over the same 5-seed span.
10. Extra online teacher shaping did not improve on top of that fix, so the best current combat setup is teacher pretraining plus active-relay credit assignment, with `combat_teacher_shaping_delta=0.0`.
11. `deadly_corridor` remains unsolved in this setup; corridor combat likely needs a stronger navigation/cover teacher, not just attack alignment.
12. The stateful corridor teacher is a partial improvement: with `pretrain_episodes=4`, `deadly_corridor` survival improved from `0.40` to `0.50` and mean damage dropped from `101.4` to `93.3` on the 5-seed batch.
13. That same corridor change did not improve mean kills, so the remaining gap is not pure cover behavior anymore; it is converting corridor positioning into reliable finishing shots.
14. The explicit attack-window transfer branch did not help. Keeping it behind CLI flags is fine for future experiments, but the best current corridor setup still leaves both attack-window deltas at `0.0`.
15. The decoder-side attack gate is the first corridor change that improved the overall policy profile after positioning transfer: on 5 seeds it kept mean kills flat at `0.8` while improving survival from `0.30` to `0.45` and tripling attack-window fire rate.
16. For `deadly_corridor`, the current best balanced setup is `pretrain_episodes=4`, `combat_decoder_attack_bonus=1.0`, `combat_decoder_attack_penalty=1.5`, with both Hebbian attack-window deltas still at `0.0`.

## Recording Artifact

Recorded GIF from the seed-42 tuned teacher-pretraining run:
- `doom_teacher_tuned_health_gathering_seed42.gif`

Older replay-based GIF:
- `doom_teacher_replay_health_gathering_seed42.gif`

Recorded combat GIF from the tuned `defend_the_center` seed-42 run:
- `doom_teacher_tuned_defend_the_center_seed42.gif`

Recorded combat GIF from the active-relay `defend_the_center` seed-42 run:
- `doom_teacher_active_relay_defend_the_center_seed42.gif`

Recorded `deadly_corridor` GIF from the `pretrain_episodes=4` seed-45 run:
- `doom_teacher_deadly_corridor_pretrain4_seed45.gif`

Recorded `deadly_corridor` GIF from the decoder-bias seed-46 run:
- `doom_teacher_deadly_corridor_decoder_bias_seed46.gif`

Raw PNG frames are in:
- `../doom_videos/doom_exp1_ep1/`
- `../doom_videos/doom_exp1_defend_the_center_ep1/`
