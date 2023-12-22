from abc import ABC, abstractmethod

"""
This file containts the abstract 'solver' class from which all solvers inherit / derive.
"""

class solver(ABC):

    """
    Generic solver class to serve as [the] mother class to all solvers supported by this package.
    """

    def __init__(self, step_size: float) -> None:

        """
        
        """

        self.step_size = step_size;

        return;