"""RLlib callbacks that publish lightweight Diep gameplay metrics and eval videos."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Mapping

from RL_training.rewards import REWARD_FIELDS

from observability.config import ObservabilityConfig
from observability.core.stats_bridge import EpisodeStatsSummary
from observability.video.eval_video import maybe_write_eval_video

try:  # RLlib is an optional dependency for observability unit tests.
    from ray.rllib.algorithms.callbacks import DefaultCallbacks
except ImportError:  # pragma: no cover - used only when Ray is not installed.
    class DefaultCallbacks:  # type: ignore[no-redef]
        """Fallback base so this module remains importable without Ray."""

        pass

logger = logging.getLogger(__name__)


def _empty_reward_totals() -> dict[str, float]:
    """Build the per-agent accumulator used while an RLlib episode is in flight."""
    return {field: 0.0 for field in REWARD_FIELDS}


def _episode_custom_data(episode: Any) -> dict[str, Any]:
    """Return Ray 2.55's mutable per-episode custom data mapping."""
    data = getattr(episode, "custom_data", None)
    if data is None:
        data = {}
        setattr(episode, "custom_data", data)
    return data


def _latest_infos(episode: Any) -> dict[str, dict[str, Any]]:
    """Read only the latest info dictionaries from Ray 2.55's episode API."""
    get_infos = getattr(episode, "get_infos", None)
    if not callable(get_infos):
        return {}

    call_attempts = (
        lambda: get_infos(-1, env_steps=True, return_list=False),
        lambda: get_infos(-1, return_list=False),
        lambda: get_infos(-1),
    )
    values: Any = None
    for call in call_attempts:
        try:
            values = call()
            break
        except TypeError:
            continue

    if isinstance(values, Mapping):
        if all(isinstance(value, Mapping) for value in values.values()):
            return {str(agent): dict(info or {}) for agent, info in values.items()}
        return {"agent_0": dict(values)}
    return {}


def _unwrap_env(env: Any) -> Any:
    """Unwrap common RLlib/PettingZoo wrappers to find the Diep parallel environment."""
    current = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "_sim") and hasattr(current, "episode_stats_array"):
            return current
        for attr in ("par_env", "pettingzoo_env", "env", "unwrapped"):
            candidate = getattr(current, attr, None)
            if candidate is not None and candidate is not current:
                current = candidate
                break
        else:
            break
    return current


def _env_from_kwargs(kwargs: Mapping[str, Any]) -> Any:
    """Extract the first concrete environment from RLlib callback keyword arguments."""
    for key in ("env", "base_env", "env_runner", "worker"):
        candidate = kwargs.get(key)
        if candidate is None:
            continue
        if key == "base_env" and hasattr(candidate, "get_sub_environments"):
            envs = candidate.get_sub_environments()
            if envs:
                return _unwrap_env(envs[0])
        for attr in ("env", "base_env"):
            nested = getattr(candidate, attr, None)
            if nested is not None:
                return _unwrap_env(nested)
        return _unwrap_env(candidate)
    return None


def _log_metric(metrics_logger: Any, key: str, value: Any) -> None:
    """Emit one metric through RLlib's Ray 2.55 MetricsLogger."""
    if metrics_logger is None:
        return
    log_value = getattr(metrics_logger, "log_value", None)
    if not callable(log_value):
        return
    log_value(key, value)


def _log_reward_metrics(metrics_logger: Any, prefix: str, sums: Mapping[str, float], count: int) -> None:
    """Emit reward component sums and means through RLlib metrics logging."""
    denominator = max(int(count), 1)
    for field in REWARD_FIELDS:
        total = float(sums.get(field, 0.0))
        _log_metric(metrics_logger, f"{prefix}/{field}_sum", total)
        _log_metric(metrics_logger, f"{prefix}/{field}_mean", total / denominator)


def _log_game_metrics(metrics_logger: Any, summary: EpisodeStatsSummary) -> None:
    """Emit episode stat fields from the native stats row through RLlib metrics logging."""
    for key, value in {
        "game/score_total": summary.score_total,
        "game/score_from_farming": summary.score_from_farming,
        "game/score_from_pvp": summary.score_from_pvp,
        "game/damage_dealt": summary.damage_dealt,
        "game/enemy_damage_dealt": summary.enemy_damage_dealt,
        "game/damage_taken": summary.damage_taken,
        "game/enemy_kills": summary.enemy_kills,
        "game/farm_kills": summary.farm_kills,
        "game/shots_fired": summary.shots_fired,
        "game/shots_hit": summary.shots_hit,
        "game/hit_rate": summary.hit_rate,
        "game/death_count": summary.death_count,
        "game/death_cause": summary.death_cause,
        "game/level_reached": summary.level_reached,
    }.items():
        _log_metric(metrics_logger, key, value)


class DiepRLlibObservabilityCallback(DefaultCallbacks):
    """Collect Diep custom metrics for RLlib/Tune and periodically emit eval videos."""

    def __init__(self, config: ObservabilityConfig | None = None):
        super().__init__()
        self.config = config or ObservabilityConfig.from_env()
        self.config.ensure_directories()
        self._started_at = time.perf_counter()
        self._last_video_iteration = 0

    def on_episode_step(self, *, episode, **kwargs):  # noqa: D401 - RLlib callback signature.
        """Accumulate reward components from latest infos exposed by RLlib."""
        data = _episode_custom_data(episode)
        reward_sums = data.setdefault("diep_reward_sums", defaultdict(_empty_reward_totals))
        normalized_sums = data.setdefault("diep_normalized_reward_sums", defaultdict(_empty_reward_totals))
        step_counts = data.setdefault("diep_step_counts", defaultdict(int))
        for agent, info in _latest_infos(episode).items():
            if agent not in self.config.stats_log_agents:
                continue
            components = info.get("reward_components") or {}
            normalized = info.get("reward_components_normalized") or {}
            if components:
                for field in REWARD_FIELDS:
                    reward_sums[agent][field] += float(components.get(field, 0.0))
                    normalized_sums[agent][field] += float(normalized.get(field, 0.0))
                step_counts[agent] += 1

    def on_episode_end(self, *, episode, metrics_logger=None, **kwargs):  # noqa: D401 - RLlib callback signature.
        """Flush accumulated stats into RLlib MetricsLogger at episode end."""
        data = _episode_custom_data(episode)
        agent = self.config.learner_agent
        reward_sums = data.get("diep_reward_sums", {}).get(agent, _empty_reward_totals())
        normalized_sums = data.get("diep_normalized_reward_sums", {}).get(agent, _empty_reward_totals())
        steps = int(data.get("diep_step_counts", {}).get(agent, 0))
        _log_reward_metrics(metrics_logger, "reward", reward_sums, steps)
        _log_reward_metrics(metrics_logger, "reward_normalized", normalized_sums, steps)

        env = _env_from_kwargs(kwargs)
        sim = getattr(env, "_sim", None)
        if sim is None:
            return
        try:
            agent_index = int(agent.split("_", 1)[1])
            rows = sim.episode_stats_array()
            total_reward = float(getattr(episode, "total_reward", 0.0) or 0.0)
            summary = EpisodeStatsSummary.from_row(
                rows[agent_index],
                episode_id=str(getattr(episode, "id_", getattr(episode, "episode_id", "episode"))),
                controlled_agent=agent,
                episode_length=steps,
                total_reward=total_reward,
            )
            _log_game_metrics(metrics_logger, summary)
        except Exception:  # pragma: no cover - metrics must not break training.
            logger.exception("Failed to collect Diep episode stats")

    def on_train_result(self, *, algorithm, metrics_logger=None, result=None, **kwargs):  # noqa: D401 - RLlib callback signature.
        """Run occasional evaluation videos after training results are available."""
        if not isinstance(result, dict):
            return
        iteration = int(result.get("training_iteration") or 0)
        if iteration <= 0 or self.config.video_interval_iterations <= 0:
            return
        if iteration % self.config.video_interval_iterations != 0 or iteration == self._last_video_iteration:
            return
        self._last_video_iteration = iteration
        started_at = time.perf_counter()
        try:
            video = maybe_write_eval_video(algorithm, self.config, iteration=iteration)
            elapsed = video.elapsed_seconds if video is not None else time.perf_counter() - started_at
            result["gameplay/eval_video_elapsed_seconds"] = elapsed
            result["gameplay/eval_video_policy_fallback"] = bool(video.used_policy_fallback) if video is not None else False
            if video is not None:
                result["gameplay/eval_video_path"] = str(video.path)
                _log_metric(metrics_logger, "gameplay/eval_video_elapsed_seconds", elapsed)
                _log_metric(metrics_logger, "gameplay/eval_video_policy_fallback", int(video.used_policy_fallback))
                try:
                    import wandb  # type: ignore

                    result["gameplay/eval_video"] = wandb.Video(str(video.path), fps=self.config.video_fps, format="mp4")
                except ImportError:
                    result["gameplay/eval_video"] = str(video.path)
        except Exception:  # pragma: no cover - video failures should never stop learning.
            logger.exception("Failed to write Diep eval video at iteration %s", iteration)
            result["gameplay/eval_video_error"] = "failed"
            result["gameplay/eval_video_elapsed_seconds"] = time.perf_counter() - started_at


__all__ = ["DiepRLlibObservabilityCallback"]
