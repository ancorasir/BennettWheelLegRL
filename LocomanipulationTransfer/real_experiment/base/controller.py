from abc import abstractmethod
import numpy as np
# from base import Base
from base.base import Base

class Controller(Base):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def retrieveInfo(self) -> np.array:
        pass

    @abstractmethod
    def command(self, cmd:np.array) -> bool:
        pass
