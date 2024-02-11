from pynamics.models._model import model
from ._simulator import simulation
import numpy as np
import pandas as pd

"""

"""

class SystemSim(simulation):

    """
    
    """

    def __init__(self, system: model, input_signal: np.ndarray, sim_options: dict = ..., solver: str = "RK4", solver_options: dict = ...) -> None:

        """
        
        """

        super().__init__(system, sim_options, solver, solver_options);
        self._input_checks(input_signal);
        self.inputs = input_signal;
        self.outputs = np.zeros(shape=(self.system.output_dim, self.time.shape[0]));

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

    def summary(self) -> None:

        """
        
        """

        return;

    def _step(self) -> np.ndarray:

        """
        
        """



        return;

    def run(self) -> pd.DataFrame:

        """
        
        """

        pass

class ControlSim(simulation):

    """
    
    """

    def __init__(self, system: model, sim_options: dict = ..., solver: str = "RK4", solver_options: dict = ...) -> None:

        """
        
        """

        super().__init__(system, sim_options, solver, solver_options);

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

class TimedControlSim(simulation):

    """
    
    """

    def __init__(self, system: model, sim_options: dict = ..., solver: str = "RK4", solver_options: dict = ...) -> None:

        """
        
        """

        super().__init__(system, sim_options, solver, solver_options);

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

# EstimatorSim -> to see how a state estimator behaves?
# Sim without any reference -> the inputs change, and the controllers are present, but no reference is passed (useful for when the goal is e.g. maximising power); 
#    Hybrids are needed -> Custom sims for this sort of systems (e.g. wind turbine sim)? 