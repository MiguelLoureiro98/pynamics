from abc import ABC, abstractmethod
import numpy as np

"""
This file contains the model base class, which forms the template for every plant model supported by this package.
"""

class model(ABC):

    """
    This is the parent class for every plant model supported by Pynamics. 
    While custom models are supported, they must all inherit from this class.

    Attributes
    ----------------------------------------------------------------------------------

    Methods
    ---------------------------------------------------------------------------------- 
    """

    def __init__(self, initial_state: np.ndarray) -> None:
        
        """
        Class constructor. Receives the system's initial state as an input.

        Arguments
        ----------------------------------------------------------------------------------
        initial_state: np.ndarray
        The system's initial state. Should be an array shaped (n, 1), where
        n is the number of variables.

        Returns
        ----------------------------------------------------------------------------------
        None
        """

        super().__init__();
        self.x = initial_state;

        return;
    
    @abstractmethod
    def get_state(self) -> np.ndarray:

        """
        
        """

        pass

    @abstractmethod
    def get_output(self) -> np.ndarray:

        """
        
        """

        pass

    @abstractmethod
    def get_input(self) -> np.ndarray:

        """
        
        """

        pass

    @abstractmethod
    def set_input(self, u: np.ndarray | float) -> None:

        """
        
        """

        pass

    @abstractmethod
    def update_state(self) -> None:

        """
        
        """

        pass

    @abstractmethod
    def eval(self) -> np.ndarray:

        """
        
        """

        pass