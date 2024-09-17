#   Copyright 2024 Miguel Loureiro

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""
This module contains the simulator base class.
"""

from abc import ABC, abstractmethod
from .models.base import BaseModel
from ._solvers._fixed_step._fixed_step_solver import _FixedStepSolver
from ._solvers._fixed_step._fixed_step_solvers import _Euler, _Modified_Euler, _Heun, _RK4
import numpy as np
import pandas as pd

class _BaseSimulator(ABC):
    """
    Simulator base class.
    """

    def __init__(self, system: BaseModel, t0: float=0.0, tfinal: float=10.0, solver: str="RK4", step_size: float=0.001) -> None:
        """
        Class constructor.
        """

        super().__init__();
        self._model_check(system);
        self._system = system;

        sim_options = {"t0": t0, "tfinal": tfinal};
        solver_options = {"t0": t0, "step_size": step_size};

        self._check_options(sim_options, solver_options);

        self._options = sim_options;
        self._solver = self._solver_selection(solver, solver_options);
        self._time = np.arange(self.options["t0"], self.options["tfinal"] + solver_options["step_size"], solver_options["step_size"]);

        return;

    def _model_check(self, system: BaseModel) -> None:
        """
        Perform a type check on the model.
        """

        if(isinstance(system, BaseModel) is False):

            raise TypeError("'system' must be an instance of the 'BaseModel' class.");

        return;

    def _check_options(self, sim_options: dict, solver_options: dict) -> None:
        """
        Perform type and value checks on every sim and solver option.
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
    
        if(sim_options["t0"] != solver_options["t0"]): # This test isn't needed anymore!!! Remove ASAP.

            raise ValueError("The initial time instant must be the same for both the simulation and the solver.\n \
                             To solve this issue, set both solver_options['t0'] and sim_options['t0'] to the same value (it must be a float).");

        return;

    def _solver_selection(self, solver: str, solver_options: dict) -> _FixedStepSolver:
        """
        Perfom a type check on `solver` and select the appropriate solver.
        """

        if (isinstance(solver, str) is False):

            raise TypeError("'solver' must be a string.");

        solvers = {"Euler": _Euler(solver_options["step_size"], solver_options["t0"]),
                   "Modified_Euler": _Modified_Euler(solver_options["step_size"], solver_options["t0"]),
                   "Heun": _Heun(solver_options["step_size"], solver_options["t0"]),
                   "RK4": _RK4(solver_options["step_size"], solver_options["t0"])};
        
        if (solver not in solvers):

            raise ValueError("The selected solver is currently not supported by this package.\n \
                             Please select one of the following: Euler, Modified_Euler, Heun, RK4.");

        return solvers[solver];

    @property
    def system(self) -> BaseModel:
        """
        Get system.
        """

        return self._system;

    @property
    def options(self) -> dict:
        """
        Get simulation options.
        """

        return self._options;

    @property
    def solver(self) -> _FixedStepSolver:
        """
        Get solver.
        """

        return self._solver;

    @property
    def time(self) -> np.ndarray:
        """
        Get time array.
        """

        return self._time;

    @abstractmethod
    def summary(self) -> None:
        """
        Display simulation options.
        """

        pass

    @abstractmethod
    def _step(self) -> any:
        """
        Perfom a simulation step.
        """

        pass

    @abstractmethod
    def run(self) -> pd.DataFrame:
        """
        Run an entire simulation.
        """

        pass