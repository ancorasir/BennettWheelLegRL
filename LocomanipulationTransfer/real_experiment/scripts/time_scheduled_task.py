import sys, os
import warnings

# root_dir = os.path.dirname(os.path.dirname(__file__))
root_dir = "/home/bionicdl/SHR/OverconstrainedRobot/OmniverseGym/experiments"
# base_dir = os.path.join(root_dir, 'base')
# ctrl_dir = os.path.join(root_dir, 'controllers')
# task_dir = os.path.join(root_dir, 'tasks')
config_dir = os.path.join(root_dir, 'cfg')
# sys.path.append(base_dir)
# sys.path.append(task_dir)
# sys.path.append(ctrl_dir)

import yaml
import argparse
import time

from tasks.quadruped_forward_locomotion_task import QuadrupedForwardLocomotion
from tasks.single_module_ik import SingleModuleIK

task_map = {
    "QuadrupedForwardLocomotion": QuadrupedForwardLocomotion,
    "SingleModuleIK": SingleModuleIK,
}

def currentTime(ms=True):
    if ms:
        return round(time.time()*1000, 2)
    else:
        return time.time()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse task name and config. ")
    parser.add_argument('task_name', type=str, help="name of the task")
    parser.add_argument('config_name', type=str, help="name of the config file")

    args = parser.parse_args()
    
    task_name = args.task_name
    if '.yaml' not in args.config_name:
        config_name = args.config_name + ".yaml"
    else:
        config_name = args.config_name
    config_path = os.path.join(config_dir, config_name)

    # read cfg file and initialize the task object
    task = task_map.get(task_name)
    assert task is not None, 'No task found with name ' + task_name
    with open(config_path, 'r') as config_f:
        cfgs = yaml.safe_load(config_f)
    task = task(cfgs)

    # Get time_interval from cfg
    time_interval_ms = cfgs.get('time_interval_ms')
    timer_sleep_time = cfgs.get('timer_sleep_seconds', 0.0001)

    # initialze the task
    task.initialize()
    last_exec_time = 0

    while task.running:
        if currentTime() - last_exec_time >= time_interval_ms:
            # Update the last execution time
            last_exec_time = currentTime()

            # Step the task
            task.step()

            # Check overdue
            time_elasped = currentTime() - last_exec_time
            if time_elasped > time_interval_ms:
                warnings.warn("Task step execution time ({} ms) is larger than the expected {} ms. ".format(time_elasped, time_interval_ms))
        else:
            time.sleep(0.0001)

    task.stop()