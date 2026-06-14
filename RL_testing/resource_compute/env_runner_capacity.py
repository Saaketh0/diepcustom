"""Probe and cache safe num_envs_per_env_runner for Diep RLlib training."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from RL_training import DiepCustomParallelEnv
from DiepModelConfig import DiepCatalog, DiepConfig, DiepPolicy
from rewards import training_env_config

try:
    import psutil
except ImportError:  # pragma: no cover - optional at runtime
    psutil = None

ENV_NAME = "diepcustom_headless"
DEFAULT_NUM_ENVS_PER_ENV_RUNNER = 4
DEFAULT_CANDIDATES = (1, 2, 4, 8, 16)
DEFAULT_MEMORY_LIMIT = 0.85

CAPACITY_FILE = Path(__file__).resolve().parent / "training_data" / "env_runner_capacity.json"

MAIN_POLICIES = ["main_class_A", "main_class_B", "main_class_C", "main_class_D"]
GHOST_POLICIES = [f"class_{c}_ghost_{i}" for c in "ABCD" for i in range(4)]

DiepRLSpec = RLModuleSpec(
    module_class=DiepPolicy,
    model_config=DiepConfig,
    catalog_class=DiepCatalog,
)


def _policy_mapping_fn(agent_id, episode, worker, **kwargs):
    index = int(agent_id.split("_")[-1])
    if index < 4:
        return MAIN_POLICIES[index]
    char_class = "ABCD"[index % 4]
    ghost_slot = (index // 4) % 4
    return f"class_{char_class}_ghost_{ghost_slot}"


def _register_diep_env():
    register_env(ENV_NAME, lambda cfg: ParallelPettingZooEnv(DiepCustomParallelEnv(**cfg)))


def _build_diep_ppo_config(num_env_runners, num_envs_per_env_runner, compute_resources):
    return (
        PPOConfig()
        .environment(ENV_NAME, env_config=training_env_config())
        .framework(framework="torch")
        .multi_agent(
            policy_mapping_fn=_policy_mapping_fn,
            policies=set(MAIN_POLICIES + GHOST_POLICIES),
            policies_to_train=MAIN_POLICIES,
        )
        .env_runners(
            num_env_runners=num_env_runners,
            num_cpus_per_env_runner=1,
            num_envs_per_env_runner=num_envs_per_env_runner,
        )
        .learners(
            num_learners=compute_resources[1],
            num_gpus_per_learner=compute_resources[2],
        )
        .resources(num_gpus=compute_resources[3])
        .rl_module(rl_module_spec=DiepRLSpec)
    )


def get_num_envs_per_env_runner(compute_resources) -> int:
    """Probe env-runner capacity and return the best num_envs_per_env_runner."""
    _register_diep_env()
    result = benchmark_and_save(
        lambda num_envs: _build_diep_ppo_config(1, num_envs, compute_resources)
    )
    return int(result["num_envs_per_env_runner"])


def load_num_envs_per_env_runner(default: int = DEFAULT_NUM_ENVS_PER_ENV_RUNNER) -> int:
    """Return cached probe result, or ``default`` if no cache exists."""
    if not CAPACITY_FILE.exists():
        return default
    try:
        data = json.loads(CAPACITY_FILE.read_text())
        return int(data.get("num_envs_per_env_runner", default))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def save_capacity_result(result: dict) -> Path:
    CAPACITY_FILE.parent.mkdir(parents=True, exist_ok=True)
    CAPACITY_FILE.write_text(json.dumps(result, indent=2) + "\n")
    return CAPACITY_FILE


def _memory_fraction() -> float:
    if psutil is None:
        return 0.0
    return psutil.virtual_memory().percent / 100.0


def _sampled_steps(train_result: dict) -> float:
    for key in (
        "num_env_steps_sampled_lifetime",
        "num_env_steps_sampled",
        "sampler_results",
    ):
        value = train_result.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, dict):
            for nested_key in ("num_env_steps_sampled_lifetime", "num_env_steps_sampled"):
                nested = value.get(nested_key)
                if isinstance(nested, (int, float)):
                    return float(nested)
    return 0.0


def probe_num_envs_per_env_runner(
    build_config: Callable[[int], object],
    *,
    candidates: Iterable[int] = DEFAULT_CANDIDATES,
    memory_limit: float = DEFAULT_MEMORY_LIMIT,
) -> dict:
    """Find the largest ``num_envs_per_env_runner`` that completes one train step."""
    trials: list[dict] = []
    best: dict | None = None

    for num_envs in candidates:
        algo = None
        trial = {"num_envs_per_env_runner": num_envs, "ok": False}
        try:
            if psutil is not None and _memory_fraction() >= memory_limit:
                trial["error"] = "memory_limit_already_reached"
                trials.append(trial)
                break

            config = build_config(num_envs)
            algo = config.build()
            start = time.perf_counter()
            train_result = algo.train()
            elapsed = max(time.perf_counter() - start, 1e-6)
            peak_memory = _memory_fraction()
            steps = _sampled_steps(train_result)
            steps_per_second = steps / elapsed if steps > 0 else 1.0 / elapsed

            trial.update(
                {
                    "ok": True,
                    "elapsed_sec": round(elapsed, 3),
                    "steps_per_second": round(steps_per_second, 3),
                    "peak_memory_fraction": round(peak_memory, 4),
                }
            )
            trials.append(trial)

            if peak_memory >= memory_limit:
                trial["error"] = "memory_limit_exceeded"
                break

            best = trial
        except Exception as exc:  # pragma: no cover - runtime probe path
            trial["error"] = repr(exc)
            trials.append(trial)
            break
        finally:
            if algo is not None:
                algo.stop()

    if best is None:
        best = next((t for t in trials if t.get("ok")), trials[-1] if trials else {})
        if not best.get("ok"):
            best = {"num_envs_per_env_runner": DEFAULT_NUM_ENVS_PER_ENV_RUNNER, "ok": True, "fallback": True}

    result = {
        "num_envs_per_env_runner": int(best.get("num_envs_per_env_runner", DEFAULT_NUM_ENVS_PER_ENV_RUNNER)),
        "benchmarked_at": datetime.now(timezone.utc).isoformat(),
        "trials": trials,
    }
    if best.get("steps_per_second") is not None:
        result["steps_per_second"] = best["steps_per_second"]
    if best.get("peak_memory_fraction") is not None:
        result["peak_memory_fraction"] = best["peak_memory_fraction"]
    return result


def benchmark_and_save(
    build_config: Callable[[int], object],
    *,
    candidates: Iterable[int] = DEFAULT_CANDIDATES,
    memory_limit: float = DEFAULT_MEMORY_LIMIT,
) -> dict:
    result = probe_num_envs_per_env_runner(
        build_config,
        candidates=candidates,
        memory_limit=memory_limit,
    )
    save_capacity_result(result)
    return result


def run_diep_capacity_probe(compute_resources, *, candidates=None, memory_limit=DEFAULT_MEMORY_LIMIT) -> dict:
    """Run the Diep env-runner probe and save results."""
    _register_diep_env()
    return benchmark_and_save(
        lambda num_envs: _build_diep_ppo_config(1, num_envs, compute_resources),
        candidates=candidates or DEFAULT_CANDIDATES,
        memory_limit=memory_limit,
    )
