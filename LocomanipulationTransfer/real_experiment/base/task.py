from abc import abstractmethod
from base.base import Base
#from time_slicer import TimeSlicer

class Task(Base):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def step(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

# class TimeScheduledTask(Task):
#     def __init__(self, time_interval_ms=30) -> None:
#         super().__init__()

#         self.time_interval_ms = time_interval_ms
#         self.time_slicer = TimeSlicer(time_interval_ms=self.time_interval_ms)

#     def startTask(self):
#         self.time_slicer.setTargetFunc(self.step)
#         self.time_slicer.start()

#     def stoptask(self):
#         self.time_slicer.stop()