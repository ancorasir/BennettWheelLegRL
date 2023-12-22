class QuadrupedTrajectory(object):
    # TRAJ_FILE_PATH = os.path.join(env_dir, "utils", "test_utils", "quadruped_forward_locomotion_traj.txt")

    def __init__(self, traj_file_path, repeat_traj=True, skip_num=0) -> None:
        self.TRAJ_FILE_PATH = traj_file_path
        self.repeat_traj = repeat_traj
        self.file_reader = open(self.TRAJ_FILE_PATH, "r")
        self.traj = []
        self.skip_num = skip_num

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
            if self.repeat_traj:
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

if __name__ == "__main__":
    traj = QuadrupedTrajectory()
    while True:
        print(traj.get_current_traj())