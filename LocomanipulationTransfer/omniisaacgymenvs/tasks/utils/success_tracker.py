import torch

class SuccessTracker:
    def __init__(self, 
                 rot_thresh=0.15, 
                 consecutive_success_steps=15,
                 max_reset_counts=2048) -> None:
        '''
            Args:
                max_reset_counts: for how many resets we calculate the sucess rate, i.e., if 
                the value is 2048, it means when we have 2048 finished environments, we calculate
                the success by num_success_envs/2048
        '''
        self.successes = None
        self.consecutive_successes = None
        self.goal_reset_buf = None

        self.max_reset_counts = max_reset_counts
        self.num_successes = None
        self.num_resets = None
        self.success_rate = None

        self.rot_thresh = rot_thresh
        self.consecutive_success_steps = consecutive_success_steps

        # Default params
        self._num_envs = 1
        self._device = "cuda"

    def init_success_tracker(self, num_envs, device):
        self._num_envs = num_envs
        self._device = device

        self.successes = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        self.consecutive_successes = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        self.goal_reset_buf = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)

        # Metrics
        # Success for both envs
        self.max_reset_counts = torch.tensor(2048, dtype=torch.long, device=self._device)
        self.num_successes = torch.tensor(0, dtype=torch.long, device=self._device)
        self.num_resets = torch.tensor(0, dtype=torch.long, device=self._device)
        self.success_rate = torch.tensor(0.0, dtype=torch.float32, device=self._device)

    def reset_bookkeeping(self, env_ids):
        num_reset = len(env_ids)
        self.successes[env_ids] = torch.zeros(num_reset, dtype=torch.long, device=self._device)
        self.consecutive_successes[env_ids] = torch.zeros(num_reset, dtype=torch.long, device=self._device)
        self.goal_reset_buf[env_ids] = torch.zeros(num_reset, dtype=torch.long, device=self._device)

    def track_success(self, 
                      current_rot_dist):
        consecutive_goal_reset = torch.where(self.consecutive_successes > self.consecutive_success_steps, 
                                                torch.ones_like(self.consecutive_successes), 
                                                torch.zeros_like(self.consecutive_successes))
        
        # Check if success
        successes = torch.where(torch.abs(current_rot_dist) <= self.rot_thresh, 
                            torch.ones_like(consecutive_goal_reset), 
                            torch.zeros_like(consecutive_goal_reset))
        
        # Check if consecutive success
        ## 1 if and only current success and last success is 1; otherwise 0
        this_consecutive_success = torch.logical_and(successes, self.successes)
        # Reset consecutive success buf if 0 
        self.consecutive_successes[:] = torch.where(this_consecutive_success == 0, torch.zeros_like(self.consecutive_successes), self.consecutive_successes)
        # Add one success to consecutive success buf if 1 (both last and current step are successful)
        self.consecutive_successes[:] = torch.where(this_consecutive_success == 1, self.consecutive_successes+1, self.consecutive_successes)
        # If last not successful, but current successful, set buf to 1 (first success)
        last_fail = torch.where(self.successes == 0, torch.ones_like(self.successes), torch.zeros_like(self.successes))
        current_success = torch.where(successes == 1, torch.ones_like(successes), torch.zeros_like(successes))
        last_fail_current_success = torch.logical_and(last_fail, current_success)
        self.consecutive_successes[:] = torch.where(last_fail_current_success == 1, torch.ones_like(self.consecutive_successes), self.consecutive_successes)
        # Update successes buf
        self.successes[:] = successes

        self.goal_reset_buf = consecutive_goal_reset

        return consecutive_goal_reset
    
    def update_success_rate(self, reset_buf):
        # Calculate the success rate
        if self.num_resets > self.max_reset_counts:
            # Update the success rate
            if self.num_resets != 0:
                self.success_rate = self.num_successes/self.num_resets
            else:
                self.success_rate = torch.tensor(0.0)

            # Zero reset counts periodically
            self.num_resets = torch.tensor(0, dtype=torch.long, device=self._device)
            self.num_successes = torch.tensor(0, dtype=torch.long, device=self._device)

        self.num_successes += torch.sum(self.goal_reset_buf)
        self.num_resets += torch.sum(reset_buf)

        return self.success_rate
