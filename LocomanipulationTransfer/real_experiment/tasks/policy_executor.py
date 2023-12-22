from inspect import stack
import os, sys
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(root_dir, "base"))
from task import Task

from rlg_policy_executor import RLGPolicyExecutor
import yaml
import numpy as np
import torch

class GymPolicyExecutor(Task):
    def __init__(self, train_config, trained_pth, num_actions, num_obs) -> None:
        super().__init__()

        self.train_config = train_config
        self.trained_pth = trained_pth
        self.num_actions = num_actions
        self.num_obs = num_obs

        self.rlg_policy_executor = None

    def initialize(self):
        self.rlg_policy_executor = RLGPolicyExecutor(self.train_config,
                                                     self.trained_pth,
                                                     self.num_actions,
                                                     self.num_obs,
                                                     device="cuda")
        self.logInfo("[Policy Executor] RLG Policy Executor ready. ")

class QuadrupedForwardPolicy(GymPolicyExecutor):
    def __init__(self, train_config, task_config, trained_pth, quadruped_controller, imu_controller, ) -> None:
        num_actions = 12
        num_obs = 46
        super().__init__(train_config, trained_pth, num_actions, num_obs)

        self.task_config = task_config

    def initialize(self):
        return super().initialize()



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
        time_start = time.time()
        print(pe.getAction(input))
        print(time.time()-time_start)

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


