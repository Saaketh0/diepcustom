# PettingZoo RL Quickstart (Python Only)

Requires **Python 3.12+** (`diepcustom/pyproject.toml`). Older interpreters fail on modern type syntax in `RL_training/` (for example `tuple[int, ...] | None`).

Use the root `RL_training/` package for Python-side agents, rewards, and PettingZoo training.
You should not need to browse C++ or conformance internals for normal reward experiments.

## Essential files

`RL_training/pettingzoo_env.py`
- Main PettingZoo `ParallelEnv` wrapper.
- Use `DiepCustomParallelEnv` in training scripts.
- Handles parallel agents, action dicts, observations, rewards, infos, reset, and step.

`RL_training/rewards.py`
- Python reward configuration and reusable reward components.
- Important symbols: `RewardConfig`, `make_reward_config`, `reward_components`.
- Add new reusable reward fields here only when simple weights are not enough.

`RL_training/actions.py`
- Converts Python trainer actions into simulator actions.
- Dict form: `{'move': [x, y], 'aim': [x, y], 'buttons': [fire, alt_fire], 'stat_upgrade_choice': i, 'tank_upgrade_choice': j}`.
- Flat form: `[move_x, move_y, aim_x, aim_y, fire, alt_fire, stat_upgrade_choice, tank_upgrade_choice]`.
- Use `-1` for either upgrade field when no upgrade is requested.

`RL_training/spaces.py`
- Gymnasium/PettingZoo action and observation spaces.
- Includes tiny fallbacks for smoke tests.

`RL_training/headless.py`
- Lower-level Python simulator wrapper.
- Use directly only for custom fast loops outside PettingZoo.

`RL_training/agents.py`
- Profile-driven multi-agent helpers.
- Use `AgentProfile` + `AgentRoster` when each env agent needs its own build/controller config.

`RL_training/__init__.py`
- Convenience exports so scripts can import from `RL_training`.

## Minimal environment

```python
from RL_training import DiepCustomParallelEnv

env = DiepCustomParallelEnv(
    seed=1,
    agents=2,
    max_ticks=1000,
    observation_mode='combat',
    include_snapshot_info=False,
    reward_config={
        'score_delta': 1.0,
        'alive': 0.01,
        'death': -1.0,
        'step': -0.001,
    },
)
```

## Basic loop

```python
observations, infos = env.reset(seed=1)

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
```

## Reward fields

`RewardConfig` supports: `raw`, `score_delta`, `health_delta`, `damage_taken`, `alive`, `death`, `truncation`, and `step`.

Tune weights in your training script: `env.set_reward_config(score_delta=2.0, death=-2.0, step=-0.001)`.

Debug components with: `infos[agent]['reward_components']`.

## Observation mode

Only `observation_mode='combat'` is supported.

It returns `{'grid_obs': ..., 'self_obs': ..., 'prev_action_obs': ...}` and is the policy-facing observation used by the RLlib combat training stack.

## RLlib training

Production training uses Ray RLlib PPO in `RL_testing/ray_code.py` (20 agents: 4 mains + 16 ghosts). See:

- [RL_testing/ghost_model.md](./RL_testing/ghost_model.md) — league loop, Redis/SSD persistence, resume requirements
- [docs/headless-pettingzoo-api.md](./docs/headless-pettingzoo-api.md) — RLlib quickstart commands

```bash
cd RL_testing
./start_redis.sh
PYTHONPATH=.. python -m league_initialization.seed_league_cache   # first time
PYTHONPATH=.. python ray_code.py
PYTHONPATH=.. python resume_from_checkpoint.py
```

Training data persists under `diepcustom/training_data/` (league weights in `redis/`, RLlib checkpoints in `RLlib/`).

## Fast training defaults

```python
DiepCustomParallelEnv(
    observation_mode='combat',
    include_snapshot_info=False,
    fast_reward_state=True,  # used by ray_code.py DIEP_ENV_CONFIG
    reward_config={'score_delta': 1.0, 'alive': 0.01, 'death': -1.0},
)
```

## Python validation files

`conformance/headless/python_pettingzoo_smoke.py`: quick env/reward check.

`conformance/headless/python_pettingzoo_api_test.py`: PettingZoo `parallel_api_test` compliance.

`conformance/headless/python_gym_combat_wrapper_smoke.py`: combat env smoke without external RL frameworks.

`conformance/headless/python_training_benchmark.py`: Python training throughput benchmark.

## Practical workflow

1. Import `DiepCustomParallelEnv` from `RL_training`.
2. Start with `observation_mode='combat'`.
3. Tune `reward_config` in your training script.
4. Inspect `reward_components` when debugging.
5. Run smoke/API tests before long training runs.
