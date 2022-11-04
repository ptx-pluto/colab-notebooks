from abc import ABC, abstractmethod
import numpy as np


class DynamicSystem(ABC):

    @abstractmethod
    def eom(self, t, y) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def ydim(self) -> int:
        pass
