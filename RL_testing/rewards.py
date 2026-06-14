"""Barebones reward configuration for RLlib Diep training."""

# Keep this intentionally simple: it is passed directly into
# DiepCustomParallelEnv(reward_config=...) and interpreted by RL_training.rewards.
BASIC_REWARD_CONFIG = {
    # Use per-step score/points deltas as the primary signal, not raw simulator events.
    "score_delta": 1.0,
    # Keep raw exposed for tuning, but disabled by default.
    "raw": 0.0,
    # Tiny living/stalling cost so doing nothing forever is not neutral.
    "step": -0.001,
    # Small explicit terminal penalty on death, in addition to raw C++ death events.
    "death": -1.0,
    # Avoid double-counting health; use an explicit damage-taken penalty instead.
    "health_delta": 0.0,
    "damage_taken": -0.01,
    # Kill/farm/level shaping nudges progress without overpowering score deltas.
    "enemy_kills": 2.0,
    "farm_kills": 0.05,
    "level_delta": 0.02,
    "level_milestone": 0.5,
    # Position/motion/combat micro-shaping is deliberately small and dense.
    "edge_proximity": -0.01,
    "movement_speed": 0.005,
    "retreat": 0.03,
    "aim_accuracy": 0.05,
    "enemy_damage_dealt": 0.02,
    # Step/death already cover survival pressure; truncation is neutral.
    "alive": 0.0,
    "truncation": 0.0,
}


def training_env_config(**overrides):
    """Return the default env config for long-running RL training."""

    config = {
        "agents": 20,
        "max_ticks": 2000,
        "scenario": "training-ffa-easy",
        "include_snapshot_info": False,
        "fast_reward_state": True,
        "normalize_reward_components": True,
        "reward_config": BASIC_REWARD_CONFIG,
    }
    config.update(overrides)
    return config


__all__ = ["BASIC_REWARD_CONFIG", "training_env_config"]
