from abc import ABC, abstractmethod
import numpy as np

"""
This file containts the abstract 'fixed step solver' class from which all fixed step solvers inherit / derive.
"""

class fixed_step_solver(ABC):

    """
    Generic fixed step solver class to serve as [the] mother class to all fixed step solvers supported by this package.
    """

    def __init__(self, step_size: float, t0: float=0.0) -> None:

        """
        
        """

        super().__init__();
        self.h = step_size;
        self.t = t0;

        return;

    def update_time_step(self) -> None:

        """
        
        """

        self.t += self.h;

        return; 

    def get_time_step(self) -> None:

        """
        
        """

        return self.t;

    @abstractmethod
    def step(self) -> tuple:

        """
        
        """

        pass