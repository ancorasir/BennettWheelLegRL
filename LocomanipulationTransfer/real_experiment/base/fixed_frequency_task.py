import time
from abc import abstractmethod

class FixedFrequencyTask:
    def __init__(self, freq_hz=50, episode_len=500) -> None:
        self.freq_hz = freq_hz
        self.episode_len = episode_len
        
        self.__dt_ms = round(1000.0/self.freq_hz)
        # self.__task_duration_second = self.episode_len * self.__dt_ms
        self.__current_progress_step = 0
        # self.__task_duration_ms = round(self.__task_duration_second * 1000.0)
        self.__start_time_ms = 0.0
        self.__last_exec_time = 0.0

    @property
    def current_time_ms(self):
        return round(time.time()*1000.0, 2)
    
    def run_task(self):

        self.__start_time_ms = self.current_time_ms
        while self.__current_progress_step <= self.episode_len:
            self.__last_exec_time = self.current_time_ms
            if self.__last_exec_time - self.__start_time_ms >= self.__dt_ms * (self.__current_progress_step):
                # Get current timestamp
                current_timestamp = self.__last_exec_time - self.__start_time_ms

                # Execute task
                self.task(current_timestamp, self.__current_progress_step)

                # Update progress step
                self.__current_progress_step += 1

            else:
                time.sleep(0.00005)

        # Close the task; clean up some staff
        self.close_task()

    @abstractmethod
    def task(self, current_timestamp_ms, current_progress_step):
        '''
            Args:
                current_timestamp: the current time in milliseconds, indicating the time elasped from the start of the task;
                current_progress_step: the current number of steps in integer, elasped from the start of the task;
        '''
        raise NotImplementedError
    
    @abstractmethod
    def close_task(self):
        pass