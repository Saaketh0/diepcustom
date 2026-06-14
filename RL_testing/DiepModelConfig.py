import torch
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (DefaultPPOTorchRLModule)
from RPPO_pipeline import RPPOInput
from torch import nn
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ACTOR, CRITIC, ENCODER_OUT

# Catalog
import functools
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.configs import MLPEncoderConfig



ENCODER_DIM = 312


def model_config_get(model_config, key, default=None):
    if isinstance(model_config, dict):
        return model_config.get(key, default)
    return getattr(model_config, key, default)


class DiepPolicy(DefaultPPOTorchRLModule):
    def setup(self):
        super().setup()
        self.encoder = Wrapper(
            observation_space=self.observation_space,
            action_space=self.action_space,
            model_config=self.model_config,
        )


class Wrapper(nn.Module):
    def __init__(self, observation_space=None, action_space=None, model_config=None) -> None:
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.model_config = model_config

        self.rppo = RPPOInput(
            observation_space=self.observation_space,
            action_space=self.action_space,
        )

        lstm_size = model_config_get(model_config, "lstm_cell_size", 256)
        self.lstm = nn.LSTM(input_size=ENCODER_DIM, hidden_size=lstm_size, batch_first=True)

    def get_initial_state(self):
        return {
            "h": torch.zeros(self.lstm.num_layers, self.lstm.hidden_size),
            "c": torch.zeros(self.lstm.num_layers, self.lstm.hidden_size),
        }

    def forward(self, batch):
        features = self.rppo(batch[Columns.OBS])
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if features.dim() == 2:
            features = features.unsqueeze(1)

        batch_size = features.shape[0]
        state_in = batch.get(Columns.STATE_IN)
        if state_in is None:
            initial = self.get_initial_state()
            h = initial["h"].unsqueeze(1).expand(-1, batch_size, -1).to(features.device)
            c = initial["c"].unsqueeze(1).expand(-1, batch_size, -1).to(features.device)
        else:
            h = state_in["h"].transpose(0, 1).to(features.device)
            c = state_in["c"].transpose(0, 1).to(features.device)

        lstm_out, (h_out, c_out) = self.lstm(features, (h, c))
        embeddings = lstm_out[:, -1, :]

        return {
            ENCODER_OUT: {ACTOR: embeddings, CRITIC: embeddings},
            Columns.STATE_OUT: {"h": h_out.transpose(0, 1), "c": c_out.transpose(0, 1)},
        }


DiepConfig = {
    # Plain custom RLModule config. Ray's DefaultModelConfig is only intended
    # for RLlib's built-in default modules; this module supplies its own
    # encoder and only needs these PPO head/recurrent settings.
    "vf_share_layers": True,
    "max_seq_len": 10,
    "lstm_cell_size": 256,
    "head_fcnet_hiddens": [],
    "head_fcnet_activation": "relu",
}


class DiepCatalog(PPOCatalog):
    def _determine_components_hook(self):
        # Don't call super() — that tries to parse Dict obs and fails.

        cfg = self._model_config_dict
        latent = cfg.get("lstm_cell_size", 312)

        self.latent_dims = [latent]

        # Dummy encoder config — only so PPOCatalog.__init__ can finish.
        # DiepPolicy.setup() replaces the built encoder with your Wrapper.
        self._encoder_config = MLPEncoderConfig(
            input_dims=[latent],
            hidden_layer_dims=[],
            output_layer_dim=latent,
        )

        self._action_dist_class_fn = functools.partial(
            self._get_dist_cls_from_action_space,
            action_space=self.action_space,
        )
