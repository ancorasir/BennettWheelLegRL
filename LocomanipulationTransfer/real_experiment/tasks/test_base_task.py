import os, sys
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(root_dir)

from base.fixed_frequency_task import FixedFrequencyTask

class TestFixedFreqTask(FixedFrequencyTask):
    def __init__(self) -> None:
        super().__init__(40, 100)

    def task(self, current_timestamp_ms, current_progress_step):
        print("Current timestamp: {}; Current progress step: {}; ".format(current_timestamp_ms, current_progress_step))


if __name__ == "__main__":
    test_task = TestFixedFreqTask()
    test_task.run_task()