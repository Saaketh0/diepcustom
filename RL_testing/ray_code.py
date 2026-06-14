from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from RL_training import DiepCustomParallelEnv
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
import ray
from league_initialization import LeagueBootstrapCallback
from league_initialization.paths import RLLIB_CHECKPOINT_DIR
from ray.tune import RunConfig, CheckpointConfig
from resource_compute import compute_resource, get_num_envs_per_env_runner
from DiepModelConfig import DiepPolicy, DiepConfig, DiepCatalog
from rewards import training_env_config
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from observability.config import ObservabilityConfig
from observability.logging.rllib_callbacks import DiepRLlibObservabilityCallback
from observability.logging.wandb_tune import create_wandb_logger_callback

ray.init()

TRAINING_ENV_CONFIG = training_env_config()
observability_config = ObservabilityConfig.from_env(eval_env_config=TRAINING_ENV_CONFIG)

register_env("diepcustom_headless", lambda cfg: ParallelPettingZooEnv(DiepCustomParallelEnv(**cfg)))


MAIN_POLICIES = ["main_class_A", "main_class_B", "main_class_C", "main_class_D"]
GHOST_POLICIES = [f"class_{c}_ghost_{i}" for c in "ABCD" for i in range(4)]

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    index = int(agent_id.split("_")[-1])
    if index < 4:
        return MAIN_POLICIES[index]
    char_class = "ABCD"[index % 4]
    ghost_slot = (index // 4) % 4
    return f"class_{char_class}_ghost_{ghost_slot}"


class TrainingCallbacks(LeagueBootstrapCallback, DiepRLlibObservabilityCallback):
    """Combine ghost-league maintenance with lightweight Diep observability."""

    def __init__(self):
        """Initialize both callback parents without relying on cooperative MRO."""
        LeagueBootstrapCallback.__init__(self)
        DiepRLlibObservabilityCallback.__init__(self, config=observability_config)

    def on_train_result(self, *, algorithm, result=None, **kwargs):
        """Refresh the league, then attach observability artifacts to the result."""
        LeagueBootstrapCallback.on_train_result(self, algorithm=algorithm, result=result, **kwargs)
        DiepRLlibObservabilityCallback.on_train_result(self, algorithm=algorithm, result=result, **kwargs)


# Calls function to get avail compute resources for training to utilize to the max
compute_resources = compute_resource()
num_envs_per_env_runner = get_num_envs_per_env_runner(compute_resources)

DiepRLSpec = RLModuleSpec(
    module_class = DiepPolicy,
    model_config = DiepConfig,
    catalog_class = DiepCatalog
)

config = (
    PPOConfig()
    # Takes in the custom environment we made above and set a agetns and max time limit
    .environment("diepcustom_headless", env_config=TRAINING_ENV_CONFIG)
    # Runs on PyTorch under the hood
    .framework(framework='torch')
    # Maps the policies from the 20 agents to the respective policies, function is above
    .multi_agent(
        policy_mapping_fn=policy_mapping_fn,
        policies=set(MAIN_POLICIES + GHOST_POLICIES),
        policies_to_train=MAIN_POLICIES,
    )

    # Configures how many parallel instances and how much CPU cores are given to each
    .env_runners(
        num_env_runners=compute_resources[0],
        num_cpus_per_env_runner=1, # 1 CPU per parallel thread worker
        # Maximize further: Vectorize multiple independent environments inside EACH worker thread
        num_envs_per_env_runner=num_envs_per_env_runner
    )
    # Computes the gradient descent that env_runners passes on
    .learners(
        num_learners=compute_resources[1],
        num_gpus_per_learner=compute_resources[2]
    )

    # How much resources to handle (idk lowk)
    .resources(num_gpus=compute_resources[3])

    # the actual Recurrent part of the RPPO
    .rl_module(rl_module_spec = DiepRLSpec)
    .callbacks(TrainingCallbacks)
)

tuner = tune.Tuner(
    "PPO",
    param_space = config,
    run_config=RunConfig(           # 3. Macro Runtime Operations
            name="rl_run",
            storage_path=str(RLLIB_CHECKPOINT_DIR),
            checkpoint_config=CheckpointConfig(
                checkpoint_frequency=5, 
                num_to_keep=50,
                checkpoint_at_end=True   
            ),
            stop={"training_iteration": 2000},
            callbacks=[create_wandb_logger_callback(observability_config)],
        )

)

tuner.fit()
