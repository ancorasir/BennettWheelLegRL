from base.fixed_frequency_task import FixedFrequencyTask

class QuadrupedTrajectoryLoader(object):
    def __init__(self, 
                 traj_file, 
                 repeat=False, 
                 skip_num=0) -> None:
        self.traj_file_path = traj_file
        self.file_reader = open(self.traj_file_path, "r")
        self.traj = []
        self.skip_num = skip_num
        self.repeat = repeat

        self.init_traj()
        self.num_traj = len(self.traj[0])
        self.current_traj_i = 0

    def init_traj(self):
        """
            Read all the lines from the file
            retrieve the data and save to self.traj
        """
        lines = self.file_reader.readlines()
        for line_i, line in enumerate(lines):
            single_traj_list = []

            # Segment a line by ,
            line_striped = line.strip().split(',')
            # Convert to float
            for position in line_striped:
                single_traj_list.append(float(position))

            self.traj.append(single_traj_list)

    def get_current_traj(self):
        if self.current_traj_i > self.num_traj - 1:
            if self.repeat:
                self.current_traj_i = 0
            else:
                return []
        
        # Get the target position for each dof
        dof_positions = []
        for i in range(len(self.traj)):
            dof_positions.append(self.traj[i][self.current_traj_i])

        # Increment trajectory pointer
        self.current_traj_i += self.skip_num + 1

        return dof_positions

    def close(self):
        self.file_reader.close()

class QuadrupedTrajectoryTracker(FixedFrequencyTask):
    def __init__(self, trajectory_file, quadruped_controller, output_file_path, freq_hz=50, episode_len=500) -> None:
        self.trajectory_file = trajectory_file
        self.trajectory_loader = QuadrupedTrajectoryLoader(self.trajectory_file)
        
        self.output_file_path = output_file_path
        self.output_file = open(self.output_file_path, "a+")
        self.output_file.write("timestamp, progress_step\n")

        self.quadruped_controller = quadruped_controller

        super().__init__(freq_hz, episode_len)

    def task(self, current_timestamp_ms, current_progress_step):
        # Get current position targets
        current_joint_position_targets = self.trajectory_loader.get_current_traj()
        if len(current_joint_position_targets) > 0:
            # Still have joint position trajectories
            # Set joint position targets
            self.quadruped_controller.set_joint_position_targets(current_joint_position_targets)
            # Read info from quadruped encoders
            self.quadruped_controller.update_joint_states()
            joint_positions = self.quadruped_controller.joint_positions
            joint_velocities = self.quadruped_controller.joint_velocities

            self.output_file.write("{},{}\n".format(current_timestamp_ms, current_progress_step))

    def close_task(self):
        self.output_file.close()
        self.trajectory_loader.close()

            
