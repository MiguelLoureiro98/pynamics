from ._model import model
import numpy as np

class linearModel(model):

    """
    This class implements a generic linear state-space model. Its methods allow one to ... .
    """

    def __init__(self, initial_state: np.ndarray, initial_control: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> None:
        
        """
        
        """

        super().__init__(initial_state)
        self.A = A;
        self.B = B;
        self.C = C;
        self.D = D;
        self.u = initial_control;

        return;

    def get_state(self) -> np.ndarray:
        
        """
        
        """

        return self.x;

    def get_output(self) -> np.ndarray:
        
        """
        
        """

        return self.C * self.x;

    def get_input(self) -> np.ndarray:

        """
        
        """

        return self.u;

    def set_input(self, u: np.ndarray | float) -> None:

        """
        
        """

        self.u = u;
    
        return;

    def update_state(self, state: np.ndarray) -> None:

        """
        
        """

        self.x = state;
    
        return;

    def eval(self, t: float, x: np.ndarray) -> np.ndarray:

        """
        
        """

        return self.A * x + self.B * self.u;