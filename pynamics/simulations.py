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

from .models._model import model
from ._controllers._dummy import dummy_controller
from ._simulator import simulation
from ._noise._noise_generators import white_noise
import numpy as np
import pandas as pd

"""
This module provides several classes for simulating dynamical systems.
Both open-loop and closed-loop simulations are supported.
"""

class sim(simulation):
    
    """
    _summary_

    _extended_summary_

    Parameters
    ----------
    simulation : _type_
        _description_
    """

    def __init__(self, system: model, input_signal: np.ndarray, t0: float=0.0, tfinal: float=10.0, solver: str="RK4", step_size: float=0.001, \
                 mode: str="open_loop", controller: any=None, reference_labels: list[str] | None=None, reference_lookahead: int=1, \
                 noise_power: int | float=0.0, noise_seed: int=0) -> None:
        
        """
        _summary_

        _extended_summary_

        Parameters
        ----------
        system : model
            _description_

        input_signal : np.ndarray
            _description_

        t0 : float, optional
            _description_, by default 0.0

        tfinal : float, optional
            _description_, by default 10.0

        solver : str, optional
            _description_, by default "RK4"

        step_size : float, optional
            _description_, by default 0.001

        controller : any, optional
            _description_, by default None

        reference_labels : list[str] | None, optional
            _description_, by default None

        reference_lookahead : int, optional
            _description_, by default 1

        noise_power : int | float, optional
            _description_, by default 0.0

        noise_seed : int, optional
            _description_, by default 0
        """

        super().__init__(system, t0, tfinal, solver, step_size);
        self._input_checks(input_signal);
        self.inputs = self._input_reformatting(input_signal);
        #self.states = np.zeros(shape=(self.system.state_dim, self.time.shape[0]));
        self.outputs = np.zeros(shape=(self.system.output_dim, self.time.shape[0]));
        self.noise = white_noise(self.system.output_dim, self.time.shape[0], noise_power, noise_seed);
        self._lookahead_check(reference_lookahead);
        self.ref_lookahead = reference_lookahead;
        self.controller = controller;

        self._mode_check(mode);

        if(mode == "open_loop"):

            self.controller = dummy_controller(self.inputs.shape[0], self.system.input_dim, step_size);

        self.control_actions = np.zeros(shape=(self.controller.output_dim, self.time.shape[0]));
        self.ref_labels = self._labels_check(reference_labels);

        return;

    def _lookahead_check(self, ref_lookahead: int) -> None:

        """
        _summary_

        _extended_summary_

        Parameters
        ----------
        ref_lookahead : int
            _description_

        Raises
        ------
        TypeError
            _description_

        ValueError
            _description_
        """

        if(isinstance(ref_lookahead, int) is False):

            raise TypeError("The 'ref_lookahead' parameter must be an integer.");
    
        if(ref_lookahead < 1):

            raise ValueError("The 'ref_lookahead' parameter must not be smaller than 1.");

        return;

    def _mode_check(self, mode: str) -> None:
        
        """
        _summary_

        _extended_summary_

        Parameters
        ----------
        mode : str
            _description_

        Raises
        ------
        TypeError
            _description_

        ValueError
            _description_

        Returns
        -------
        None
        """

        if(isinstance(mode, str) is False):

            raise TypeError("'mode' should be a string.");
    
        else:

            if(mode != "open_loop" and mode != "closed_loop"):

                raise ValueError("Please select either 'open_loop' or 'closed_loop' as simulation mode.");

        return;

    def _labels_check(self, labels: list[str] | None) -> list[str]:

        """
        
        """

        if(labels is None):

            new_labels = [f"Ref_{num}" for num in range(1, self.inputs.shape[0] + 1)];
        
        else:

            if(isinstance(labels, list) is False):

                raise TypeError("'reference_labels' must be a list.");
    
            elif(len(labels) != self.inputs.shape[0]):

                raise ValueError("The number of reference labels must be the same as the number of reference signals.");
    
            new_labels = labels;

        return new_labels;

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

    def _input_reformatting(self, input_signal: np.ndarray) -> np.ndarray:

        """
        
        """

        input_shape_length = len(input_signal.shape);

        if(input_shape_length == 1):

            input_signal = np.expand_dims(input_signal, axis=0);

        return input_signal;

    def summary(self) -> None:

        """
        _summary_

        _extended_summary_
        """

        return;

    def _step(self, t: float, ref: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        
        """
        _summary_

        _extended_summary_

        Parameters
        ----------
        t : float
            _description_

        ref : np.ndarray
            _description_

        y : np.ndarray
            _description_

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            _description_
        """

        if(t % self.controller.Ts == 0):

            control_actions = self.controller.control(ref, y); # This should only be done when t % Ts = 0 (i.e. if this instant happens to coincide with a sampling instant)
            self.system.set_input(control_actions);
        
        else:

            control_actions = self.system.get_input();
        
        new_state = self.solver.step(self.system);
        self.system.update_state(new_state); # Add time -> it is important for time varying systems (is it really?)
        outputs = self.system.get_output();

        return (outputs, control_actions);

    def run(self) -> pd.DataFrame:
        
        """
        _summary_

        _extended_summary_

        Returns
        -------
        pd.DataFrame
            Data frame containing the results.
        """

        print("-----------------------------------------------------------------");
        print("Simulation details...");
        print("-----------------------------------------------------------------");

        self.control_actions[:, 0] = self.system.get_input();
        self.outputs[:, 0] = self.system.get_output();

        #for ind, (t, _, y, n, _) in enumerate(zip(self.time[:-1], self.inputs[:, :-1], self.outputs[:, :-1], self.noise[:, :-1], self.control_actions[:, :-1])):
        for ind, t in enumerate(self.time[:-1]):

            #self.outputs[:, ind+1], self.control_actions[:, ind+1] = self._step(t, ref, y + n);
            self.outputs[:, ind+1], self.control_actions[:, ind+1] = self._step(t, self.inputs[:, ind:ind+self.ref_lookahead], self.outputs[:, ind:ind+1] + self.noise[:, ind:ind+1]);
            #self.outputs[:, ind+1], self.control_actions[:, ind+1] = self._step(t, self.inputs[:, ind:ind+self.ref_lookahead], y + n);
            self.outputs[:, ind+1] += self.noise[:, ind+1];
        
        # Create results data frame
        names = ["Time"];
        names.extend(self.ref_labels);
        names.extend(self.system.input_labels);
        names.extend(self.system.output_labels);

        results = np.expand_dims(self.time, axis=0).T;
        results = np.hstack((results, self.inputs.T));
        results = np.hstack((results, self.control_actions.T));
        results = np.hstack((results, self.outputs.T));

        sim_data = pd.DataFrame(results, columns=names);

        return sim_data;

class SIL(simulation):

    """
    
    """

    def __init__(self, system: model, t0: float = 0, tfinal: float = 10, solver: str = "RK4", step_size: float = 0.001) -> None:

        """
        
        """

        super().__init__(system, t0, tfinal, solver, step_size);

        raise NotImplementedError("SIL simulations have yet to be implemented.");

class RLSim(simulation):

    """
    
    """

    def __init__(self, system: model, t0: float=0.0, tfinal: float=10.0, solver: str="RK4", step_size: float=0.001) -> None:

        """
        
        """

        super().__init__(system, t0, tfinal, solver, step_size);

        raise NotImplementedError("RL simulations haven't been implemented yet.");

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