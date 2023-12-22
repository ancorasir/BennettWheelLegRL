import torch

# Import the skrl components to build the RL system
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env
from skrl.envs.torch import load_omniverse_isaacgym_env
from skrl.utils import set_seed

# set the seed for reproducibility and deterministic training
set_seed(42, deterministic=True)
torch.use_deterministic_algorithms(mode=True)
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

from networks.mlp import SharedMLP

# Load and wrap the Omniverse Isaac Gym environment
# env = load_omniverse_isaacgym_env(task_name="HorizontalForwardWalking")
env = load_omniverse_isaacgym_env(task_name="WalkToTarget")
env = wrap_env(env)

class DataRecordPPO(PPO):
    def _update(self, timestep: int, timesteps: int) -> None:
        _task = env._env.unwrapped._task
        log_dict = _task.extras
        for key_name in log_dict.keys():
            self.track_data(key_name, float(log_dict[key_name].item()))
        return super()._update(timestep, timesteps)

device = env.device

eval = True

# Instantiate a RandomMemory as rollout buffer (any memory can be used for this)
if eval:
    memory = RandomMemory(memory_size=4800, num_envs=env.num_envs, device=device)
else:
    memory = RandomMemory(memory_size=48, num_envs=env.num_envs, device=device)

# Instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#spaces-and-models
models_ppo = {}
models_ppo["policy"] = SharedMLP(env.observation_space, env.action_space, device)
models_ppo["value"] = models_ppo["policy"]  # same instance: shared model

# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#configuration-and-hyperparameters
cfg_ppo = PPO_DEFAULT_CONFIG.copy()
if eval:
    cfg_ppo["rollouts"] = 4800  # memory_size
else:
    cfg_ppo["rollouts"] = 48  # memory_size

cfg_ppo["learning_epochs"] = 5
cfg_ppo["mini_batches"] = 1  # 24 * 4096 / 32768
cfg_ppo["discount_factor"] = 0.99
cfg_ppo["lambda"] = 0.95
cfg_ppo["learning_rate"] = 3e-4
cfg_ppo["learning_rate_scheduler"] = KLAdaptiveRL
cfg_ppo["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.016}
cfg_ppo["random_timesteps"] = 0
cfg_ppo["learning_starts"] = 0
cfg_ppo["grad_norm_clip"] = 1.0
cfg_ppo["ratio_clip"] = 0.2
cfg_ppo["value_clip"] = 0.2
cfg_ppo["clip_predicted_values"] = True
cfg_ppo["entropy_loss_scale"] = 0.0
cfg_ppo["value_loss_scale"] = 1.0
cfg_ppo["kl_threshold"] = 0
cfg_ppo["rewards_shaper"] = None
cfg_ppo["state_preprocessor"] = RunningStandardScaler
cfg_ppo["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg_ppo["value_preprocessor"] = RunningStandardScaler
cfg_ppo["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints each 120 and 1200 timesteps respectively
cfg_ppo["experiment"]["write_interval"] = 30

env_dir = os.path.dirname(os.path.dirname(__file__))
runs_dir = os.path.join(env_dir, "runs", "ForwardWalking")
cfg_ppo["experiment"]["directory"] = runs_dir
if eval:
    cfg_ppo["experiment"]["experiment_name"] = "test"
else:
    cfg_ppo["experiment"]["experiment_name"] = "1201-omni_direction-full_penalty"
cfg_ppo["experiment"]["checkpoint_interval"] = 120

if eval:
    cfg_ppo["experiment"]["wandb"] = False
else:
    cfg_ppo["experiment"]["wandb"] = False
cfg_ppo["experiment"]["wandb_kwargs"] = {"project": "WalkToTargetRL",
                           "entity": "locomanipulation",
                           "group": "FullDirectionalWalking",
                           "name": cfg_ppo["experiment"]["experiment_name"],
                           "sync_tensorboard": True,
                           }

agent = DataRecordPPO(models=models_ppo,
                      memory=memory,
                      cfg=cfg_ppo,
                      observation_space=env.observation_space,
                      action_space=env.action_space,
                      device=device)

# # runs_root = os.path.join(env_dir, "runs")
# # checkpoint_dir = os.path.join(runs_root, "CylinderRotation", "0709-lgsk_rew-seperated_euler", "checkpoints", "best_agent.pt")

#TODO: not run from checkpoint
# checkpoint_dir = os.path.join(runs_dir, "1201-omni_direction-full_penalty", "checkpoints", "best_agent.pt")
# agent.load(checkpoint_dir)
# # agent.set_mode('eval')

# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 100000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()