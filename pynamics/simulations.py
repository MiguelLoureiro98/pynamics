from models._model import model
from ._simulator import simulation
import numpy as np
import pandas as pd

"""

"""

class sim(simulation):

    """
    
    """

    def __init__(self, system: model, input_signal: np.ndarray, t0: float=0.0, tfinal: float=10.0, solver: str="RK4", step_size: float=0.001, \
                 mode: str="open_loop", controller: any | None=None) -> None:

        """
        
        """

        super().__init__(system, t0, tfinal, solver, step_size);
        self._input_checks(input_signal);
        self.inputs = input_signal;
        self.states = np.zeros(shape=(self.system.state_dim, self.time.shape[0]));
        self.outputs = np.zeros(shape=(self.system.output_dim, self.time.shape[0]));
        self.controller = controller;
        self.control_actions = np.zeros(shape=(self.controller.output_dim, self.time.shape[0]));

        return;

    def _input_checks(self, input_signal: np.ndarray) -> None:

        """
        
        """

        if(isinstance(input_signal, np.ndarray) is False):

            raise TypeError("'input_signal' should be a Numpy array.");
    
        input_shape_length = len(input_signal.shape);
    
        if(input_shape_length == 1 and input_signal.shape[0] != self.time.shape[0] or input_shape_length != 1 and input_signal.shape[1] != self.time.shape[0]):

            raise ValueError("The input signal length must match the length of the time vector: (T_final - T_initial) / step_size. Check the number of columns of your input signal.");

        if(input_shape_length != 1 and input_signal.shape[0] != self.system.input_dim):

            raise ValueError("The input signal must have the same dimensions as the system's input. Check the number of rows of your input signal.");

        return;

    def _no_control(self, ref: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:

        """
        
        """

        return ref;

    def summary(self) -> None:

        """
        
        """

        return;

    def _step(self, t: float, ref: np.ndarray, y: np.ndarray) -> np.ndarray:

        """
        
        """

        

        return;

    def run(self) -> pd.DataFrame:

        """
        
        """

        pass

class RLSim(simulation):

    """
    
    """

    def __init__(self, system: model, t0: float=0.0, tfinal: float=10.0, solver: str = "RK4", step_size: float=0.001) -> None:

        """
        
        """

        super().__init__(system, t0, tfinal, solver, step_size);

        return;

    def summary(self) -> None:

        """
        
        """
        
        return;

    def _step(self) -> np.ndarray:
        
        """
        
        """

        pass

    def run(self) -> pd.DataFrame:

        """
        
        """

        pass

# Sim without any reference -> the inputs change, and the controllers are present, but no reference is passed (useful for when the goal is e.g. maximising power); 
#    Hybrids are needed -> Custom sims for this sort of systems (e.g. wind turbine sim)? 