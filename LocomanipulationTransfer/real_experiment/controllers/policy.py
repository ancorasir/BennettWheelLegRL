import torch
import torch.nn as nn

from skrl.agents.torch.ppo import PPO
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from gym import spaces

import numpy as np

class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 64),
                                 nn.ELU())

        self.mean_layer = nn.Linear(64, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        self.value_layer = nn.Linear(64, 1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        if role == "policy":
            return self.mean_layer(self.net(inputs["states"])), self.log_std_parameter, {}
        elif role == "value":
            return self.value_layer(self.net(inputs["states"])), {}

class Policy:
    def __init__(self,
                 checkpoint_path,
                 num_obs,
                 num_actions,
                 device="cuda") -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.num_obs = num_obs
        self.num_actions = num_actions

        models_ppo = {}
        models_ppo["policy"] = Shared(num_obs, num_actions, self.device)
        models_ppo["value"] = models_ppo["policy"]  # same instance: shared model
        
        self.gym_obs_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
        self.gym_action_space = spaces.Box(np.ones(self.num_actions) * -1.0, np.ones(self.num_actions) * 1.0)
        
        cfg_ppo = PPO_DEFAULT_CONFIG.copy()
        
        from skrl.resources.preprocessors.torch import RunningStandardScaler
        cfg_ppo["state_preprocessor"] = RunningStandardScaler
        cfg_ppo["state_preprocessor_kwargs"] = {"size": self.gym_obs_space, "device": device}
        cfg_ppo["value_preprocessor"] = RunningStandardScaler
        cfg_ppo["value_preprocessor_kwargs"] = {"size": 1, "device": device}

        self.fake_agent = PPO(models=models_ppo,  # models dict
                              cfg=cfg_ppo,  # configuration dict (preprocessors, learning rate schedulers, etc.)
                              observation_space=self.gym_obs_space,
                              action_space=self.gym_action_space,
                              device=self.device)
        
        self.fake_agent._rnn = {}
        self.fake_agent.load(self.checkpoint_path)
        
    def get_raw_actions(self, raw_env_obs):
        with torch.no_grad():
            actions, _, _ = self.fake_agent.act(raw_env_obs, 0, 0)
            actions = torch.clamp(actions, -1, 1).to(self.device).clone()
        return actions
    
    def get_actions(self, obs_np):
        obs_tensor = torch.from_numpy(obs_np).to(dtype=torch.float32, device=self.device).view(1, self.num_obs)
        action_tensor = self.get_raw_actions(obs_tensor)
        return action_tensor.clone().detach().to('cpu').numpy()[0]
        
if __name__ == "__main__":
    import numpy as np

    checkpoint_path = "/home/bionicdl/SHR/LocomanipulationTransfer/RobotLearning/omniisaacgymenvs/runs/SKRL-QuadrupedPoseControlCustomController/0526-1action_scale_rm_rot_dist_dec_rew/checkpoints\best_agent.pt"
    num_obs = 88
    num_actions = 12
    device = "cuda"

    policy = Policy(checkpoint_path,
                    num_obs,
                    num_actions,
                    device)
    
    fake_env_obs = np.ones((num_obs))
    print(policy.get_actions(fake_env_obs))
