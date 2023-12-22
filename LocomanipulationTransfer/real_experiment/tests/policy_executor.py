from inspect import stack
import os, sys
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(root_dir, "base"))
from task import Task

from gym import spaces
from rl_games.torch_runner import Runner, _restore
from rl_games.common.a2c_common import ContinuousA2CBase
from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.common.algo_observer import AlgoObserver
import yaml
import numpy as np
import torch

class FakeObs():
    def __init__(self, shape) -> None:
        self.shape = shape
        self.high = np.ones(shape)
        self.low = -1*np.ones(shape)

class RLGPolicyExecutor(Task):
    def __init__(self, train_config, trained_pth) -> None:
        super().__init__()

        self.train_config = train_config
        self.trained_pth = trained_pth

        self.rlg_continuous_a2c = None
        self.rlg_agent = None
        self.model = None
        # Load train config
        self.runner = None
        self.player = None

        self.num_actions = 12
        self.num_observations = 46
        self.num_states = 46

        with open(self.train_config, 'r') as stream:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            config = yaml.safe_load(stream)
            config['params']['seed'] = 1
            env_info = self.get_env_info()
            config['params']['config'] = {**config['params']['config'], **env_info, 'reward_shape': {}}
            # env_info = {"observation_space": FakeObs([46]), "state__space": FakeObs([46]), "action_space": FakeObs([12])}
            # _config_appending = {'features': {'observer': AlgoObserver()}, 'env_info': env_info}
            # params['config'] = {**params['config'], **_config_appending}
            config['params']['config']['device'] = device
            config['params']['config']['device_name'] = device
            # params['config']['num_actors'] = 1
            # self.rlg_continuous_a2c = A2CAgent("test", params)
            # self.rlg_continuous_a2c.restore(self.trained_pth)

            self.runner = Runner()
            self.runner.load(config)
            play_args = {
            'train': False,
            'play': True,
            'checkpoint': self.trained_pth,
            'sigma': None
            }
            self.player = self.runner.create_player()
            _restore(self.player, play_args)
            self.player.has_batch_dimension = True


            #self.model = self.rlg_continuous_a2c.config['network']
            #self.model.__init__()

    def get_env_info(self):
        info = {}
        info['action_space'] = self.action_space = spaces.Box(np.ones(self.num_actions) * -1.0, 
                                                              np.ones(self.num_actions) * 1.0)
        info['observation_space'] = spaces.Box(np.ones(self.num_observations) * -np.Inf, 
                                               np.ones(self.num_observations) * np.Inf)
        info['state_space'] = self.state_space = spaces.Box(np.ones(self.num_states) * -np.Inf, 
                                                            np.ones(self.num_states) * np.Inf)
        info['agents'] = 1

        return {'env_info': info}

    def getAction(self, obs):
        return self.player.get_action(obs, True)

if __name__ == "__main__":
    import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_config = "/home/bionicdl/SHR/OverconstrainedRobot/" + \
                   "OmniverseGym/omniisaacgymenvs/cfg/train/" + \
                   "QuadrupedForwardLocomotionPPO.yaml"
    trained_pth = "/home/bionicdl/SHR/OverconstrainedRobot/" + \
                  "OmniverseGym/omniisaacgymenvs/runs/1011-1520/" + \
                  "nn/last_1011-1520_ep_12250_rew_1496.6177.pth"

    pe = RLGPolicyExecutor(train_config, trained_pth)

    input = torch.tensor([[ 
            0.0970,  0.0179,  0.8291, -0.0654, -0.1823,  0.0794, -0.4917,  0.5302,
            0.4845, -0.4915, -0.2973, -0.6584,  0.6526,  0.2869,  0.6116,  0.2651,
            -0.2889, -0.6377, -0.0029,  0.0289, -0.0171,  0.0187, -0.0199, -0.0137,
            -0.0130,  0.0113,  0.0123,  0.0398, -0.0013, -0.0428, -0.0027, -0.0139,
            0.0169,  0.0050,  0.0393, -0.0145, -0.0245,  0.0040, -0.0293, -0.0134,
            -0.0227,  0.0239, -0.0126,  0.0022, -1.0022,  0.7390]], dtype=torch.float32, device=device)

    for i in range(1):
        #time_start = time.time()
        print(pe.getAction(input))
        #print(time.time()-time_start)

# Output from Isaac Sim
# ****
# tensor([[ 0.0970,  0.0179,  0.8291, -0.0654, -0.1823,  0.0794, -0.4917,  0.5302,
#           0.4845, -0.4915, -0.2973, -0.6584,  0.6526,  0.2869,  0.6116,  0.2651,
#          -0.2889, -0.6377, -0.0029,  0.0289, -0.0171,  0.0187, -0.0199, -0.0137,
#          -0.0130,  0.0113,  0.0123,  0.0398, -0.0013, -0.0428, -0.0027, -0.0139,
#           0.0169,  0.0050,  0.0393, -0.0145, -0.0245,  0.0040, -0.0293, -0.0134,
#          -0.0227,  0.0239, -0.0126,  0.0022, -1.0022,  0.7390]],
#        device='cuda:0')
# tensor([[-1.1873, -0.2462,  1.0345, -1.0249,  1.0088,  0.8211,  0.6503,  0.0168,
#           0.0974, -1.4583, -1.1023, -0.9538]], device='cuda:0')
# res_dict["actions"]:
# tensor([[-3.8769, -0.5473,  3.0612, -3.2072,  3.8357,  1.7266,  1.9165, -0.4235,
#          0.0089, -1.2155, -3.7891, -2.4601]], device='cuda:0')
# ****
# tensor([[ 0.0970,  0.0179,  0.8291, -0.0654, -0.1823,  0.0794, -0.4917,  0.5302,
#           0.4845, -0.4915, -0.2973, -0.6584,  0.6526,  0.2869,  0.6116,  0.2651,
#          -0.2889, -0.6377, -0.0029,  0.0289, -0.0171,  0.0187, -0.0199, -0.0137,
#          -0.0130,  0.0113,  0.0123,  0.0398, -0.0013, -0.0428, -0.0027, -0.0139,
#           0.0169,  0.0050,  0.0393, -0.0145, -0.0245,  0.0040, -0.0293, -0.0134,
#          -0.0227,  0.0239, -0.0126,  0.0022, -1.0022,  0.7390]],
#        device='cuda:0')
# tensor([[-1.1873, -0.2462,  1.0345, -1.0249,  1.0088,  0.8211,  0.6503,  0.0168,
#           0.0974, -1.4583, -1.1023, -0.9538]], device='cuda:0')


