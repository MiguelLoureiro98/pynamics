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
This module provides several classes for simulating dynamical systems.
Both open-loop and closed-loop simulations are supported.

Classes
-------
Sim
    Simulate a dynamical system and plot the results.
"""

from .models.base import BaseModel
from .controllers.dummy import DummyController
from ._simulator import _BaseSimulator
from ._noise._noise_generators import _white_noise
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Sim(_BaseSimulator):    
    """
    Simulate a dynamical system.

    This class can be used to simulate the behaviour of a dynamical system. It supports both open- \
    and closed-loop simulations, which makes it appropriate for both system analysis and control design.

    Parameters
    ----------
    system : BaseModel
        System to simulate. Must be described by a model supported by pynamics.
        ! Add link to 'pynamics' page.

    input_signal : np.ndarray
        Input signals. These may be reference values or other external inputs (e.g. wind speed in a wind turbine system).

    t0 : float, default=0.0
        Initial time instant. Must be non-negative.

    tfinal : float, optional
        ... 

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
            
    noise_seed : int, optional
        _description_, by default 0

    TODO: Attributes and Methods sections.
    """

    def __init__(self, 
                 system: BaseModel, 
                 input_signal: np.ndarray, 
                 t0: float=0.0, 
                 tfinal: float=10.0, 
                 solver: str="RK4", 
                 step_size: float=0.001, 
                 mode: str="open_loop", 
                 controller: any=None, 
                 reference_labels: list[str] | None=None, 
                 reference_lookahead: int=1, \
                 noise_power: int | float=0.0, 
                 noise_seed: int=0) -> None:
        """
        Class constructor.
        """

        super().__init__(system, t0, tfinal, solver, step_size);
        self._mode_check(mode);
        self._input_checks(input_signal, mode);
        self.inputs = self._input_reformatting(input_signal);
        self.outputs = np.zeros(shape=(self.system.output_dim, self.time.shape[0]));
        self.noise = _white_noise(self.system.output_dim, self.time.shape[0], noise_power, noise_seed);
        self._lookahead_check(reference_lookahead);
        self.ref_lookahead = reference_lookahead;
        self.controller = controller;

        if(mode == "open_loop"):

            self.controller = DummyController(self.inputs.shape[0], self.system.input_dim, step_size);

        self.control_actions = np.zeros(shape=(self.controller.output_dim, self.time.shape[0]));
        self.ref_labels = self._labels_check(reference_labels);

        return;

    def _lookahead_check(self, ref_lookahead: int) -> None:
        """
        Perform type and value checks on the `ref_lookahead` parameter.
        """

        if(isinstance(ref_lookahead, int) is False):

            raise TypeError("The 'ref_lookahead' parameter must be an integer.");
    
        if(ref_lookahead < 1):

            raise ValueError("The 'ref_lookahead' parameter must not be smaller than 1.");

        return;

    def _mode_check(self, mode: str) -> None:    
        """
        Perform type and value checks on the `mode` parameter.
        """

        if(isinstance(mode, str) is False):

            raise TypeError("'mode' should be a string.");
    
        else:

            if(mode != "open_loop" and mode != "closed_loop"):

                raise ValueError("Please select either 'open_loop' or 'closed_loop' as simulation mode.");

        return;

    def _labels_check(self, labels: list[str] | None) -> list[str]:
        """
        Perform type and value checks on the labels.
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

    def _input_checks(self, input_signal: np.ndarray, mode: str) -> None:
        """
        Perform type and value checks on the input signal (reference values).
        """

        if(isinstance(input_signal, np.ndarray) is False):

            raise TypeError("'input_signal' should be a Numpy array.");
    
        input_shape_length = len(input_signal.shape);
    
        if(input_shape_length == 1 and input_signal.shape[0] != self.time.shape[0] or input_shape_length != 1 and input_signal.shape[1] != self.time.shape[0]):

            raise ValueError("The input signal length must match the length of the time vector: (T_final - T_initial) / step_size. Check the number of columns of your input signal.");

        if(mode == "open_loop"):

            if(input_shape_length != 1 and input_signal.shape[0] != self.system.input_dim):

                raise ValueError("The input signal must have the same dimensions as the system's input. Check the number of rows of your input signal.");

        return;

    def _input_reformatting(self, input_signal: np.ndarray) -> np.ndarray:
        """
        Reformat the input signal array if need be.
        """

        input_shape_length = len(input_signal.shape);

        if(input_shape_length == 1):

            input_signal = np.expand_dims(input_signal, axis=0);

        return input_signal;

    def summary(self) -> None:
        """
        Display simulation options.

        This method can be used ... .

        TODO: Add examples section ...
        """

        print("Simulation details");
        print("------------------");

        return;

    def _step(self, t: float, ref: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  
        """
        Perform a simulation step. Used by the `run` method.
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
        Run a simulation.

        This method is used to run a simulation.

        Returns
        -------
        pd.DataFrame
            Data frame containing the results.

        TODO: Add examples section ... 
        """

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

    @staticmethod
    def tracking_plot(sim_results: pd.DataFrame,
                      time_variable: str,
                      reference: str,
                      output: str, 
                      plot_title: str="Simulation results", 
                      xlabel: str="t", 
                      ylabel: str="y", 
                      plot_height: int | float=10.0, 
                      plot_width: int | float=10.0) -> None:
        """
        _summary_

        _extended_summary_

        Parameters
        ----------
        sim_results : pd.DataFrame
            Simulation results.

        time_variable : str
            Name of the time variable.

        reference : str
            Name of the reference variable.

        output : str
            Name of the output variable.

        plot_title : str, default="Simulation results"
            Plot title.

        xlabel : str, default="t"
            X-axis label.

        ylabel : str, default="y"
            Y-axis label.

        plot_height : int | float, default=10.0
            Figure height.

        plot_width : int | float, default=10.0
            Figure width.
        """
        
        _ = plt.figure(figsize=(plot_height, plot_width));
        plt.plot(sim_results[time_variable], sim_results[reference], label="r");
        plt.plot(sim_results[time_variable], sim_results[output], label="y");
        plt.xlabel(xlabel);
        plt.ylabel(ylabel);
        plt.title(plot_title);
        plt.grid(visible=True);
        xfactor = 1.0005;
        yfactor = 1.05;
        minlim = np.fmin(sim_results[output].min(), sim_results[reference].min());
        maxlim = np.fmax(sim_results[output].max(), sim_results[reference].max());
        plt.xlim([sim_results[time_variable].min() * xfactor, sim_results[time_variable].max() * xfactor]);
        plt.ylim([minlim * yfactor, maxlim * yfactor]);
        plt.legend();
        plt.show();

        return;

    @staticmethod
    def system_outputs_plot(sim_results: pd.DataFrame,
                            time_variable: str,
                            outputs: list[str], 
                            plot_title: str="Simulation results", 
                            xlabel: str="t", 
                            ylabel: str="y", 
                            plot_height: int | float=10.0, 
                            plot_width: int | float=10.0) -> None:
        """
        _summary_

        _extended_summary_

        Parameters
        ----------
        sim_results : pd.DataFrame
            Simulation results.

        time_variable : str
            Name of the time variable.

        outputs : list[str]
            List containing the names of the output variables.

        plot_title : str, default="Simulation results"
            Plot title.

        xlabel : str, default="t"
            X-axis label.

        ylabel : str, default="y"
            Y-axis label

        plot_height : int | float, default=10.0
            Figure height.

        plot_width : int | float, default=10.0
            Figure width.
        """

        fig, axes = plt.subplots(len(outputs), 1, sharex=True);
        fig.set_figheight(plot_height);
        fig.set_figwidth(plot_width);
        fig.suptitle(plot_title);
        fig.supxlabel(xlabel);
        xfactor = 1.0005;
        yfactor = 1.05;

        if (len(outputs) > 1):

            for it, (_, output) in enumerate(zip(axes, outputs)):

                axes[it].plot(sim_results[time_variable], sim_results[output], label=output);
                axes[it].set_ylabel(ylabel);
                axes[it].grid(visible=True);
                axes[it].legend();
                axes[it].set_xlim([sim_results[time_variable].min() * xfactor, sim_results[time_variable].max() * xfactor]);
                axes[it].set_ylim([sim_results[output].min() * yfactor, sim_results[output].max() * yfactor]);
        
        else:

            axes.plot(sim_results[time_variable], sim_results[outputs[0]], label=outputs[0]);
            axes.set_ylabel(ylabel);
            axes.grid(visible=True);
            axes.legend();
            axes.set_xlim([sim_results[time_variable].min() * xfactor, sim_results[time_variable].max() * xfactor]);
            axes.set_ylim([sim_results[outputs[0]].min() * yfactor, sim_results[outputs[0]].max() * yfactor]); 

        plt.show();
    
        return;

    @classmethod
    def step_response(cls,
                      system: BaseModel, 
                      step_magnitude: int | float=1.0, 
                      t0: float=0.0, 
                      tfinal: float=10.0, 
                      solver: str="RK4", 
                      step_size: float=0.001, 
                      mode: str="open_loop", 
                      controller: any=None, 
                      reference_labels: list[str] | None=None, 
                      reference_lookahead: int=1, \
                      noise_power: int | float=0.0, 
                      noise_seed: int=0):
        """
        Simulate the step response of a dynamical system.

        This method can be used to perform a step response simulation. Keep in mind that, for now, it should only be used \
        with single-input systems or controllers needing only one reference signal.

        Returns
        -------
        Sim
            A simulation class instance.
        """

        reference_signal = np.full(shape=(1, (tfinal - t0) / step_size + 1), fill_value=step_magnitude);

        return cls(system, 
                   reference_signal, 
                   t0, 
                   tfinal, 
                   solver, 
                   step_size, 
                   mode, 
                   controller, 
                   reference_labels, 
                   reference_lookahead, 
                   noise_power, 
                   noise_seed);

    @classmethod
    def impulse_response(cls,
                         system: BaseModel, 
                         impulse_magnitude: int | float=1.0, 
                         t0: float=0.0, 
                         tfinal: float=10.0, 
                         solver: str="RK4", 
                         step_size: float=0.001, 
                         mode: str="open_loop", 
                         controller: any=None, 
                         reference_labels: list[str] | None=None, 
                         reference_lookahead: int=1, \
                         noise_power: int | float=0.0, 
                         noise_seed: int=0):
        """
        _summary_

        _extended_summary_

        Returns
        -------
        Sim
            A simulation class instance.
        """

        reference_signal = np.zeros(shape=(1, (tfinal - t0) / step_size + 1));
        reference_signal[0, 0] = impulse_magnitude;

        return cls(system, 
                   reference_signal, 
                   t0, 
                   tfinal, 
                   solver, 
                   step_size, 
                   mode, 
                   controller, 
                   reference_labels, 
                   reference_lookahead, 
                   noise_power, 
                   noise_seed);

    @classmethod
    def ramp(cls,
             system: BaseModel, 
             slope: int | float=1.0, 
             t0: float=0.0, 
             tfinal: float=10.0, 
             solver: str="RK4", 
             step_size: float=0.001, 
             mode: str="open_loop", 
             controller: any=None, 
             reference_labels: list[str] | None=None, 
             reference_lookahead: int=1, \
             noise_power: int | float=0.0, 
             noise_seed: int=0):
        """
        _summary_

        _extended_summary_

        Returns
        -------
        Sim
            A simulation class instance.
        """

        reference_signal = slope * np.arange(t0, tfinal + step_size, step_size);

        return cls(system, 
                   reference_signal, 
                   t0, 
                   tfinal, 
                   solver, 
                   step_size, 
                   mode, 
                   controller, 
                   reference_labels, 
                   reference_lookahead, 
                   noise_power, 
                   noise_seed);