from abc import ABC, abstractmethod

"""
This file containts the abstract 'fixed step solver' class from which all fixed step solvers inherit / derive.
"""

class fixed_step_solver(ABC):

    """
    Generic fixed step solver class to serve as [the] mother class to all fixed step solvers supported by this package.

    Attributes
    ----------------------------------------------------------------------------------
    h: float
    Solver step size.

    t: float
    The current solver time step.

    Methods
    ----------------------------------------------------------------------------------
    __init__
    _update_time_step
    get_time_step
    step
    """

    def __init__(self, step_size: float, t0: float=0.0) -> None:

        """
        
        """

        super().__init__();
        self.h = step_size;
        self.t = t0;

        return;

    def _update_time_step(self) -> None:

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