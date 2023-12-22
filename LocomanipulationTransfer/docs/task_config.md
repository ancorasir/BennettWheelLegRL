# Task Configuration Files

## Basic structure of task configuration files
The contents in a task configuration file mainly consist of 3 parts:

1. Isaac Sim required configs. These are the configs passed to the backend of the simulator to start up the Isaac Sim simulation.

1. Custom configs. These configs are passed to the task, to modify the task related parameters.

1. Domain randomization configs. These are the configs to set up domain randomization parameters. Detailed documentation can be found in [domain_randomization.md](https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs/blob/main/docs/domain_randomization.md)

## Isaac Sim required configs

We only explain the important parameters, most of the paramers can kept the same for all tasks.


    sim.dt: the time interval to calculate the physics. The lower, the more acurate.

    sim.max_episode_length: the maximum number of steps for an episode.

    sim.default_physics: the material properties including static and dynamic friction coefficients. If you see the friction is 2.0, this does not necessarily mean the friction coefficient between the robot tips and the ground is 2.0. The friction between two materials are calculated from averaging the two material frictions, and we have set the friciton of the ground floor to be 0.0, so the actual friction between robot and the ground floor will be half of the values set here, which is 1.

## Custom configs

All customized configs are put inside the "env" key. There are some configs that are required for all tasks, some configs are only for a specific task.
### Custom configs required by all tasks
```
  default_robot_states:
    default_joint_positions: [-1.57, 1.57, 1.57, -1.57,
                              -1.04, -2.09, 
                              2.09, 1.04, 
                              2.09, 1.04, 
                              -1.04, -2.09, 
                              1.37, -1.37, 
                              1.37, -1.37, 
                              1.37, -1.37, 
                              1.37, -1.37]
    default_base_positions: [0, 0, 0.18]
    default_base_quaternions: [1.0, 0.0, 0.0, 0.0]

  default_actuator_params:
    # It determines control freqency of the task
    control_decimal: 5 # 5 * 1/200s = 1/40s -> 40hz
    control_kp: 4.5
    control_kd: 0.2
    action_scale: 0.15
    joint_friction: 0.007
    joint_damping: 0.008
    max_torque: 1.5
```
The above configs are required for all tasks, they determine the initial robot states (the states that the robot will be set to when it falls or the episode timeouts) and the default actuator parameters such as Kp and Kd gains of the PD position controller.

```
  randomization:
    randomize_init_joint_positions: False
    rand_joint_position: 0.1 # Add random values (max to 0.2) to the initial joint angles

    randomize_init_joint_velocities: False
    rand_joint_velocities: 0.1 # Add random init joint velocities (max to 0.2)

    randomize_init_base_positions: False
    rand_xy_position: 0.1
    rand_z_position: 0.1

    randomize_init_base_quaternions: False
    rand_max_roll: 0.1
    rand_max_pitch: 0.1
    rand_max_yaw: 3.14

    # Control params randomization
    randomize_interval: 400 # Every 400 step
    randomize_friction: False
    rand_friction_range: [1.6, 2.0] # Actual friction coeff is 0.8 ~ 1.0

    randomize_control_params: False
    rand_kp_scale_range: [0.8, 1.2]
    rand_kd_scale_range: [0.8, 1.2]
    rand_action_scale_range: [0.8, 1.2]
    rand_max_torque_scale_range: [0.8, 1.2]
```
The above are also required for all tasks. They are configs used to randomize the simulation parameters (that are not supported using the official domain randomization methods). They mainly determines the material friction randomization (the official domain randomization of material friction does not work, see [How does Randomization on Material Properties Work?](https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs/issues/13) for more details), initial robot state randomizations, and joint controller randomizations.

### Task-specific configs

*Note: for all tasks by default, the order of joint names is fixed. For our four-limbed robot, we have a total of 12 joints, with 3 joints in a single limb module. We define the name of a joint to be in the form of a<module_id>_dof<joint_id>. For example, for the first module of id 1, we have 3 joints named a1_dof1, a1_dof2, a1_dof3. When we read the joint positions, velocities or torques, the values always correspond to the joints in the following order: [a1_dof1, a2_dof1, a3_dof1, a4_dof1, a1_dof2, a1_dof3, a2_dof2, a2_dof3, a3_dof2, a3_dof3, a4_dof2, a4_dof3]*

1. LocomotionInPlaceRotationControl:
  - Goal of the task:

    The task is to control the quadruped robot on a flat terrain so that the rotation of the robot matches the goal rotation;

  - Observation of the task: 
    
    The observation of the task can be customized to include extra information. The basic observation vector includes:
      
      1. Position of the terrian in robot frame (3D)
      1. Z-axis of the terrain frame expressed in robot frame (3D) 
      1. The quaternion difference between the current robot rotation and the goal robot rotation (4D)
      1. Joint positions of the actuators (12D)
      1. Joint velocities of the actuators (12D)
      1. Current actions (12D)
      1. Last actions (12D)
      1. (optional) Joint torques exerted by the actuators (12D)
      1. (optional) Linear and angular velocities of the terrain relative to the robot (6D, 3D for linear and 3D for angular velocity)
      1. (optional) Contact forces detected by the tips (4D or 12D depening on the configs set)

    Order of the observation follows the above sequence. If the optional observation is not include, just delete that observation from the above sequence. For example, if we include contact force but not joint torques and linear/angular velocities, the observation vector will be like:
      1. Position
      1. Z-axis
      1. Quaternion difference
      1. Joint positions
      1. Joint velocities
      1. Current actions
      1. Last actions
      1. Contact forces

    The optional observations can be set in the task configs:
    ```
      obs:
        include_joint_torques: True 
        
        include_velocities: False
        
        include_tip_contact_forces: True 
        full_contact_forces: True 
        binary_force_threshold: 0.5
    ```
    To include a specific observation to the observation vector, set the corresponding config to True. 
    
    For tip contact forces, two types of contact force observation can be chosen. If "full_contact_forces" is set to False, the observation will only include a binary value of 4D (each represents a single tip, and our quadruped robot has four tips), with 0 indicating the tip does not have any contact with the environment, and 1 meaning the tip contacts with the environment. Whether or not having contacts depends on the "binary_force_threshold". If the contact force magnitudes of the tip is larger than the threshold, the tip will be seen as having contacts with the environment. When "full_contact_forces" is set to True, the observation will include a 12D vector, consisting of 4 force vectors, each representing a 3D force vector detected by the tip of a limb.

    *Note: by default the contact force sensor is activated for the Isaac Sim robot. We found that if the config sim.disable_contact_processing is set to False, the simulation will be extremely slow. So be sure the param is set to True*

  - Other configs:

    ```
    indicator_position: [0.0, 0.0, 0.3] # position of the marker
    ```
    This determines the position of the marker in simulation (position of the marker will not change)
    ```
    goal_roll_range: [-0.4, 0.4]
    goal_pitch_range: [-0.4, 0.4]
    goal_yaw_range: [-1.57, 1.57]
    ```
    This determines the range of the goal rotations in Euler angles
    ```
    obs_scales:
      position_scale: 1.0
      quaternion_scale: 1.0
      up_vec_scale: 1.0
      joint_position_scale: 0.3
      joint_velocity_scale: 0.1
      # Extra observation scales
      joint_torque_scale: 0.6
      linear_vel_scale: 1.0
      angular_vel_scale: 0.2
      contact_force_scale: 0.1

    rew_scale:
      rotation_scale: 0.5
      rotation_eps: 0.1
      position_deviation_scale: -2.5
      joint_acc_scale: -0.0005
      action_rate_scale: -0.02
      success_reward: 600.0
      fall_penalty: 0.0
    ```
    The above is the observation and reward scales (the raw observation/reward term will be multiplied by the scales)

    ```
    success_conditions:
      rot_thresh: 0.15
      consecutive_success_steps: 20
    ```
    The success condition configs. The task is seen to be successfully only when the difference between the robot rotation and the goal rotation is within the rot_thresh, consecutively for 20 steps (if the difference is within the thresh for 19 steps, but in the 20th step, the difference is outside the thresh, it will not be seen as a success)
    ```
    early_termination_conditions:
      baseline_base_height: 0.05
      baseline_knee_height: 0.04
      baseline_corner_height: 0.01
    ```
    The configs used to check if the robot falls. We do not use contact sensors, we check if the heights of the base/knees/corners are below the above baseline heights to determine robot falling.

