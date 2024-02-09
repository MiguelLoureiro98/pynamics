from pynamics.models._model import model
from ._simulator import simulation
import numpy as np
import pandas as pd

"""

"""

class SystemSim(simulation):

    """
    
    """

    def __init__(self, system: model, sim_options: dict = ..., solver: str = "RK4", solver_options: dict = ...) -> None:

        """
        
        """

        super().__init__(system, sim_options, solver, solver_options);
        #self.references = np.zeros(shape=());
        #self.outputs = np.zeros(shape=());

        return;

    def step(self) -> np.ndarray:

        """
        
        """

        pass

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

    def step(self) -> np.ndarray:
        
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

    def step(self) -> np.ndarray:

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