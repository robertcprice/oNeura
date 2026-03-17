# Results Index

This directory is the checked-in artifact surface for oNeura. Treat these files
as working-project outputs rather than a frozen benchmark release.

## Start Here

- `pong_compare_20260310/`
  Current measured 25K Pong comparison across Mac, A100, and H200.
- `doom_teacher_pretraining_20260310/`
  Documented Doom teacher-pretraining sweeps and recorded artifacts.
- `pong_retina_pretraining_20260310/`
  Documented Pong retina/teacher-pretraining sweeps and notes.

## Additional Checked-In Result Bundles

- `doom_arena/`
  Arena navigation and drug-response JSON artifacts.
- `doom_combat/`
  Doom combat gameplay output and result files.
- `doom_combat_results.json`
  Top-level Doom combat summary artifact.
- `fly_ecosystem/`
  Drosophila ecosystem result JSON.
- `hybrid_ppo_deadly_corridor_20260310/`
  Hybrid PPO deadly-corridor training runs with ablations and logs.
- `hybrid_ppo_deadly_corridor_dishbrain_eval_20260310/`
  DishBrain-side evaluation runs for the hybrid PPO corridor setup.
- `pong_input_impact_20260310/`
  Pong input impact sweeps.
- `pong_retina_fe_tiebreak_20260310/`
  Retina free-energy tiebreak comparison runs.
- `pong_signal_optimization_20260310/`
  Pong signal optimization sweeps.
- `pong_speed_20260310/`
  Pong speed experiment output.
- `pong_sweep_20260310/`
  Held-out Pong sweep artifacts.

## Practical Notes

- When a result subdirectory contains its own `README.md`, start there first.
- When README text and result JSON disagree, treat the JSON/log artifact as the
  source of truth.
- Some folders contain videos, GIFs, or logs alongside JSON. Those are kept
  intentionally when they support a benchmark or demo claim.
