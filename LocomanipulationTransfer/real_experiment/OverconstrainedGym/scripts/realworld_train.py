import os, sys

gym_dir = "/home/bionicdl/SHR/OverconstrainedRobot/OmniverseGym/experiments/OverconstrainedGym"
exp_dir = os.path.dirname(gym_dir)

sys.path.append(gym_dir)
sys.path.append(exp_dir)

import torch
from rl_games.common.algo_observer import AlgoObserver
from rl_games.algos_torch import torch_ext
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from env.oc_env import OCVecEnv
from env.single_module_ik import SingleModuleIK

'''
max_current: 297
max_velocity: 42
'''

test = True
train_name = "single_module_ik_train_dof1_position"
task_cfg_path = os.path.join(gym_dir, 'cfg', 'task', 'single_module_ik.yaml')
train_cfg_path = os.path.join(gym_dir, 'cfg', 'train', 'SingleModuleIKPPO.yaml')
#checkpoint_path = os.path.join(exp_dir, 'runs', 'single_module_ik_train', 'nn', 'last_single_module_ik_train_ep_1300_rew_-272.05148.pth')
checkpoint_path = '/home/bionicdl/SHR/OverconstrainedRobot/OmniverseGym/omniisaacgymenvs/runs/1108-correct_dof_range2/nn/last_1108-correct_dof_range2_ep_900_rew_1382.8638.pth'
save_pth_dir = os.path.join(gym_dir, 'runs')


class RLGPUAlgoObserver(AlgoObserver):
    """Allows us to log stats from the env along with the algorithm running stats. """

    def __init__(self):
        pass

    def after_init(self, algo):
        self.algo = algo
        self.mean_scores = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.ep_infos = []
        self.direct_info = {}
        self.writer = self.algo.writer

    def process_infos(self, infos, done_indices):
        assert isinstance(infos, dict), "RLGPUAlgoObserver expects dict info"
        if isinstance(infos, dict):
            if 'episode' in infos:
                self.ep_infos.append(infos['episode'])

            if len(infos) > 0 and isinstance(infos, dict):  # allow direct logging from env
                self.direct_info = {}
                for k, v in infos.items():
                    # only log scalars
                    if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                        self.direct_info[k] = v

    def after_clear_stats(self):
        self.mean_scores.clear()

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.ep_infos:
            for key in self.ep_infos[0]:
                    infotensor = torch.tensor([], device=self.algo.device)
                    for ep_info in self.ep_infos:
                        # handle scalar and zero dimensional tensor infos
                        if not isinstance(ep_info[key], torch.Tensor):
                            ep_info[key] = torch.Tensor([ep_info[key]])
                        if len(ep_info[key].shape) == 0:
                            ep_info[key] = ep_info[key].unsqueeze(0)
                        infotensor = torch.cat((infotensor, ep_info[key].to(self.algo.device)))
                    value = torch.mean(infotensor)
                    self.writer.add_scalar('Episode/' + key, value, epoch_num)
            self.ep_infos.clear()
        
        for k, v in self.direct_info.items():
            self.writer.add_scalar(f'{k}/frame', v, frame)
            self.writer.add_scalar(f'{k}/iter', v, epoch_num)
            self.writer.add_scalar(f'{k}/time', v, total_time)

        if self.mean_scores.current_size > 0:
            mean_scores = self.mean_scores.get_mean()
            self.writer.add_scalar('scores/mean', mean_scores, frame)
            self.writer.add_scalar('scores/iter', mean_scores, epoch_num)
            self.writer.add_scalar('scores/time', mean_scores, total_time)

import yaml
with open(task_cfg_path, 'r') as task_cfg_f:
    task_cfg = yaml.safe_load(task_cfg_f)
with open(train_cfg_path, 'r') as train_cfg_f:
    train_cfg = yaml.safe_load(train_cfg_f)

task_env = SingleModuleIK(task_cfg)

vecenv.register('RealWorldTrain',
                lambda config_name, num_actors, **kwargs: OCVecEnv(config_name, num_actors, **kwargs))
env_configurations.register('realworld_train', {
            'vecenv_type': 'RealWorldTrain',
            'env_creator': lambda **kwargs: task_env
        })

# Change experiment name
train_cfg['params']['config']['name'] = train_name
train_cfg['params']['config']['full_experiment_name'] = train_name
runner = Runner(RLGPUAlgoObserver())
runner.load(train_cfg)

runner.reset()

runner.run({
    'train': not test,
    'play': test,
    'checkpoint': checkpoint_path,
    'sigma': None
})