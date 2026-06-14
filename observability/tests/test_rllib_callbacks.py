"""Tests for Ray 2.55-native RLlib observability callbacks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from observability.config import ObservabilityConfig
from observability.core.stats_bridge import EPISODE_STATS_FIELDS
from observability.logging import rllib_callbacks
from observability.logging.rllib_callbacks import DiepRLlibObservabilityCallback
from observability.video.eval_video import EvalVideoResult


class FakeEpisode:
    def __init__(self) -> None:
        self.custom_data: dict = {}
        self.total_reward = 12.5
        self.id_ = "fake-episode"
        self.get_infos_calls: list[tuple[tuple, dict]] = []

    def get_infos(self, *args, **kwargs):
        self.get_infos_calls.append((args, kwargs))
        if args != (-1,):
            raise AssertionError("callback must request only the latest info index")
        return {
            "agent_0": {
                "reward_components": {"score_delta": 2.0, "step": 1.0},
                "reward_components_normalized": {"score_delta": 0.5, "step": 0.25},
            },
            "agent_1": {
                "reward_components": {"score_delta": 99.0, "step": 99.0},
                "reward_components_normalized": {"score_delta": 99.0, "step": 99.0},
            },
        }


class FakeMetricsLogger:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def log_value(self, key, value, **kwargs) -> None:
        self.values[key] = value


class FakeSim:
    def episode_stats_array(self):
        values = {field: 0.0 for field in EPISODE_STATS_FIELDS}
        values.update(
            {
                "lifetime_steps": 7.0,
                "score_total": 100.0,
                "score_from_farming": 25.0,
                "score_from_pvp": 75.0,
                "damage_dealt": 9.0,
                "enemy_damage_dealt": 4.0,
                "damage_taken": 3.0,
                "shots_fired": 10.0,
                "shots_hit": 4.0,
                "enemy_kills": 2.0,
                "farm_kills": 5.0,
                "level_reached": 8.0,
            }
        )
        return np.asarray([[values[field] for field in EPISODE_STATS_FIELDS]], dtype=np.float64)


class FakeEnv:
    def __init__(self) -> None:
        self._sim = FakeSim()

    def episode_stats_array(self):
        return self._sim.episode_stats_array()


def test_rllib_callback_uses_custom_data_latest_infos_and_metrics_logger(tmp_path: Path):
    config = ObservabilityConfig(run_id="callback-test", runs_root=tmp_path, stats_log_agents=("agent_0",))
    callback = DiepRLlibObservabilityCallback(config=config)
    episode = FakeEpisode()

    callback.on_episode_step(episode=episode)
    callback.on_episode_step(episode=episode)

    assert "diep_reward_sums" in episode.custom_data
    assert episode.custom_data["diep_step_counts"]["agent_0"] == 2
    assert all(call[0] == (-1,) for call in episode.get_infos_calls)

    metrics = FakeMetricsLogger()
    callback.on_episode_end(episode=episode, metrics_logger=metrics, env=FakeEnv(), env_index=0)

    assert metrics.values["reward/score_delta_sum"] == 4.0
    assert metrics.values["reward/score_delta_mean"] == 2.0
    assert metrics.values["reward_normalized/score_delta_sum"] == 1.0
    assert metrics.values["game/score_total"] == 100.0
    assert metrics.values["game/hit_rate"] == 0.4


def test_train_result_video_metadata_and_config_propagation(monkeypatch, tmp_path: Path):
    eval_env_config = {"agents": 20, "max_ticks": 123, "seed": 9}
    config = ObservabilityConfig(
        run_id="video-test",
        runs_root=tmp_path,
        eval_env_config=eval_env_config,
        video_interval_iterations=1,
    )
    callback = DiepRLlibObservabilityCallback(config=config)
    expected_path = config.eval_iteration_dir(1) / "eval.mp4"

    def fake_maybe_write_eval_video(algorithm, observed_config, *, iteration):
        assert observed_config.eval_env_config == eval_env_config
        assert iteration == 1
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        expected_path.write_bytes(b"fake-mp4")
        return EvalVideoResult(expected_path, elapsed_seconds=0.25, used_policy_fallback=True)

    monkeypatch.setattr(rllib_callbacks, "maybe_write_eval_video", fake_maybe_write_eval_video)
    result = {"training_iteration": 1}
    metrics = FakeMetricsLogger()

    callback.on_train_result(algorithm=object(), result=result, metrics_logger=metrics)

    assert result["gameplay/eval_video_path"] == str(expected_path)
    assert result["gameplay/eval_video_elapsed_seconds"] == 0.25
    assert result["gameplay/eval_video_policy_fallback"] is True
    assert metrics.values["gameplay/eval_video_elapsed_seconds"] == 0.25
    assert metrics.values["gameplay/eval_video_policy_fallback"] == 1
