from abc import ABC, abstractmethod
import numpy as np

"""

"""

class model(ABC):

    """
    
    """

    def __init__(self, initial_state: np.ndarray) -> None:
        
        """
        
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
    def eval(self) -> np.ndarray:

        """
        
        """

        pass