from ._model import model
import numpy as np

class linearModel(model):

    """
    This class implements a generic linear state-space model. Its methods allow one to ... .
    """

    def __init__(self, initial_state: np.ndarray, initial_control: np.ndarray | float, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> None:
        
        """
        
        """

        super().__init__(initial_state)
        self.A = A;
        self.B = B;
        self.C = C;
        self.D = D;

        float_test = isinstance(initial_control, float);

        if (float_test is True):

            initial_control = np.array([initial_control]);

        array_1D_test = isinstance(initial_control, np.ndarray) is True and initial_control.shape[0] == 1;

        if (array_1D_test is True):

            initial_control = np.expand_dims(initial_control, axis=1);
        
        self.u = initial_control;

        return;

    def get_state(self) -> np.ndarray:
        
        """
        
        """

        return self.x;

    def get_output(self) -> np.ndarray:
        
        """
        
        """

        return np.matmul(self.C, self.x);

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

        return np.matmul(self.A, x) + np.matmul(self.B, self.u);