from abc import ABC, abstractmethod
from models._model import model
from solvers.fixed_step._fixed_step_solver import fixed_step_solver
from solvers.fixed_step._fixed_step_solvers import Euler, Modified_Euler, Heun, RK4
import numpy as np
import pandas as pd

"""
This file ... .
"""

class simulation(ABC):

    """
    
    """

    def __init__(self, system: model, t0: float=0.0, tfinal: float=10.0, solver: str="RK4", step_size: float=0.001) -> None:
        
        """
        
        """

        super().__init__();
        self.system = system;

        sim_options = {"t0": t0, "tfinal": tfinal};
        solver_options = {"t0": t0, "step_size": step_size};

        self._check_options(sim_options, solver_options);

        self.options = sim_options;
        self.solver = self._solver_selection(solver, solver_options);
        self.time = np.arange(self.options["t0"], self.options["tfinal"], solver_options["step_size"]);
        #self.time = np.expand_dims(self.time, axis=0);

        return;

    def _check_options(self, sim_options: dict, solver_options: dict) -> None:

        """
        
        """

        for (key, option) in solver_options.items():

            if(isinstance(option, float) is False):

                if(isinstance(option, int) is True):

                    solver_options[key] = float(option);
                
                else:

                    raise TypeError("Every solver option must be either a float or an integer.");
    
        for (key, option) in sim_options.items():

            if(isinstance(option, float) is False):

                if(isinstance(option, int) is True):

                    solver_options[key] = float(option);
                
                else:

                    raise TypeError("Every simulation option must be either a float or an integer.");
    
        if(sim_options["t0"] != solver_options["t0"]):

            raise ValueError("The initial time instant must be the same for both the simulation and the solver.\n \
                             To solve this issue, set both solver_options['t0'] and sim_options['t0'] to the same value (it must be a float).");

        return;

    def _solver_selection(self, solver: str, solver_options: dict) -> fixed_step_solver:

        """
        
        """

        if (isinstance(solver, str) is False):

            raise TypeError("'solver' must be a string.");

        solvers = {"Euler": Euler(solver_options["step_size"], solver_options["t0"]),
                   "Modified_Euler": Modified_Euler(solver_options["step_size"], solver_options["t0"]),
                   "Heun": Heun(solver_options["step_size"], solver_options["t0"]),
                   "RK4": RK4(solver_options["step_size"], solver_options["t0"])};
        
        if (solver not in solvers):

            raise ValueError("The selected solver is currently not supported by this package.\n \
                             Please select one of the following: Euler, Modified_Euler, Heun, RK4.");

        return solvers[solver];

    @abstractmethod
    def summary(self) -> None:

        """
        
        """

        pass

    @abstractmethod
    def _step(self) -> any:

        """
        
        """

        pass

    @abstractmethod
    def run(self) -> pd.DataFrame:

        """
        
        """

        pass