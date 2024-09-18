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
from .controllers.base import BaseController
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
        TODO : Add link to 'pynamics' page.

    input_signal : np.ndarray
        Input signals. These may be reference values or other external inputs (e.g. wind speed in a wind turbine system).

    t0 : float, default=0.0
        Initial time instant. Must be non-negative.

    tfinal : float, default=0.0
        Final time instant. Must be non-negative.

    solver : {"Euler", "Modified_Euler", "Heun", "RK4"}, str, default="RK4"
        Fixed-step solver.

    step_size : float, default=0.001
        Solver step size. Must be positive.

    mode : {"open_loop", "closed_loop"}, str, default="open_loop"
        Simulation mode. The controller will not be included in the simulation unless \
        parameter is set to "closed_loop".

    controller : BaseController | None, optional
        Controller.

    reference_labels : list[str] | None, optional
        List of labels for the reference signals.
            
    reference_lookahead : int, default=1
        Number of time steps ahead for which the reference values are known to the controller.
            
    noise_power : int | float, default=0.0
        White noise power. If equal to zero, no noise will be added to the simulation.

    noise_seed : int, default=0
        Random seed for the noise array.

    Attributes
    ----------
    system : BaseModel
        The system to simulate.

    options : dict
        Simulation options (initial and final time instants).

    solver : _FixedStepSolver
        Solver.

    time : np.ndarray
        Time array.

    inputs : np.ndarray
        The input signals. 

    outputs : np.ndarray
        The output signals.

    noise : np.ndarray
        Array of white noise values.

    control_actions : np.ndarray
        Array of control actions.

    controller : BaseController
        Controller.

    ref_lookahead : int
        Number of time steps ahead for which the reference values are known to the controller.

    ref_labels : list[str]
        List of labels for the reference signals.
    
    Methods
    -------
    summary()
        Display the current simulation settings.

    run()
        Run a simulation.

    reset(initial_state: np.ndarray, initial_control: np.ndarray | float)
        Reset the output and control actions arrays, as well as the system's state and input vectors, so that a new simulation can be run.

    tracking_plot(sim_results: pd.DataFrame, time_variable: str, reference: str, output: str,  plot_title: str="Simulation results",  xlabel: str="t",  ylabel: str="y",  plot_height: int | float=10.0,  plot_width: int | float=10.0)
        Evaluate the system's tracking performance by plotting the reference signal and the system's output (SISO systems only).
    
    system_outputs_plot(sim_results: pd.DataFrame, time_variable: str, outputs: list[str],  plot_title: str="Simulation results",  xlabel: str="t",  ylabel: str="y",  plot_height: int | float=10.0,  plot_width: int | float=10.0)
        Visualise the system's output signals.

    step_response(cls, system: BaseModel,  step_magnitude: int | float=1.0,  t0: float=0.0,  tfinal: float=10.0,  solver: str="RK4",  step_size: float=0.001,  mode: str="open_loop",  controller: BaseController | None=None,  reference_labels: list[str] | None=None,  reference_lookahead: int=1, \ noise_power: int | float=0.0,  noise_seed: int=0)
        Simulate the system's step response (single-input systems or single-reference controllers only).
    
    ramp(cls, system: BaseModel,  slope: int | float=1.0,  t0: float=0.0,  tfinal: float=10.0,  solver: str="RK4",  step_size: float=0.001,  mode: str="open_loop",  controller: BaseController | None=None,  reference_labels: list[str] | None=None,  reference_lookahead: int=1, \ noise_power: int | float=0.0,  noise_seed: int=0)
        Simulate the system's response to a ramp signal (single-input systems or single-reference controllers only).
    
    Raises
    ------
    TypeError
        If a value of the wrong type is passed as a parameter.

    ValueError
        If the value of any parameter is invalid (e.g. input signal has the wrong length, \
        `mode` is neither "open_loop" nor "closed_loop", etc.).

    Warning
    -------
    Only fixed-step solvers are support at the moment.
    """

    def __init__(self, 
                 system: BaseModel, 
                 input_signal: np.ndarray, 
                 t0: float=0.0, 
                 tfinal: float=10.0, 
                 solver: str="RK4", 
                 step_size: float=0.001, 
                 mode: str="open_loop", 
                 controller: BaseController | None=None, 
                 reference_labels: list[str] | None=None, 
                 reference_lookahead: int=1, 
                 noise_power: int | float=0.0, 
                 noise_seed: int=0) -> None:
        """
        Class constructor.
        """

        super().__init__(system, t0, tfinal, solver, step_size);
        self._mode_check(mode);
        self._mode = mode;
        self._input_checks(input_signal);
        self._inputs = self._input_reformatting(input_signal);
        self.outputs = np.zeros(shape=(self.system.output_dim, self.time.shape[0]));
        self.noise = _white_noise(self.system.output_dim, self.time.shape[0], noise_power, noise_seed);
        self._lookahead_check(reference_lookahead);
        self._ref_lookahead = reference_lookahead;
        self.controller = controller;

        if(self._mode == "open_loop"):

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

    def _input_checks(self, input_signal: np.ndarray) -> None:
        """
        Perform type and value checks on the input signal (reference values).
        """

        if(isinstance(input_signal, np.ndarray) is False):

            raise TypeError("'input_signal' should be a Numpy array.");
    
        input_shape_length = len(input_signal.shape);
    
        if(input_shape_length == 1 and input_signal.shape[0] != self.time.shape[0] or input_shape_length != 1 and input_signal.shape[1] != self.time.shape[0]):

            raise ValueError("The input signal length must match the length of the time vector: (T_final - T_initial) / step_size. Check the number of columns of your input signal.");

        if(self._mode == "open_loop"):

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

    @property
    def inputs(self) -> np.ndarray:
        """
        Get the input signals.

        This method can be used to access the input / reference signals using dot notation.

        Returns
        -------
        np.ndarray
            System input signals.
        """

        return self._inputs;

    @inputs.setter
    def inputs(self, new_input: np.ndarray) -> None:
        """
        Set the input signals.

        This method can be used to assign new input signals to the Sim class instance using dot notation. \
        It also performs some checks on the array and does some reformatting if necessary.

        Parameters
        ----------
        new_input : np.ndarray
            New input signals.

        Raises
        ------
        TypeError
            If `new_input` is not an np.ndarray.

        ValueError
            If `new_input` has the wrong length or its dimensions do not match those of the system's input.
        """

        self._input_checks(new_input);
        self._inputs = self._input_reformatting(new_input);

        return;

    @property
    def ref_lookahead(self) -> int:
        """
        Get the `ref_lookahead` parameter.

        This method can be used to access the value of the `ref_lookahead` parameter using dot notation.

        Returns
        -------
        int
            Number of time steps ahead for which the reference values are known to the controller.
        """

        return self._ref_lookahead;

    @ref_lookahead.setter
    def ref_lookahead(self, new_value: int) -> None:
        """
        Set the value of the `ref_lookahead` parameter.

        This method can be used to set the value of the `ref_lookahead` parameter \
        using dot notation. It also performs some checks on the new value.

        Parameters
        ----------
        new_value : int
            New value for the `ref_lookahead` parameter.

        Raises
        ------
        TypeError
            If the new value is not an integer.

        ValueError
            If the new value is smaller than one.
        """

        self._lookahead_check(new_value);
        self._ref_lookahead = new_value;

        return;

    def summary(self) -> None:
        """
        Display the current simulation settings.

        This method displays the value of the most important simulation options.

        Examples
        --------
        >>> import numpy as np
        >>> from pynamics.models.state_space_models import LinearModel
        >>> from pynamics.simulations import Sim
        >>> 
        >>> A = np.array([[0, 0, -1], [1, 0, -3], [0, 1, -3]]);
        >>> B = np.array([1, -5, 1]).reshape(-1, 1);
        >>> C = np.array([0, 0, 1]);
        >>> D = np.array([0]);
        >>> model = LinearModel(np.zeros((3, 1)), np.zeros((1, 1)), A, B, C, D);
        >>>
        >>> simulation = Sim(model, input_signal=np.ones(int(10/0.001)+1));
        >>> simulation.summary();
        Simulation settings
        -------------------
        Initial time step: 0.0 s
        Final time step: 10.0 s
        Solver step size: 0.001 s
        -------------------
        Input signals format: (1, 10001)
        Output signals format: (1, 10001)
        Control actions format: (1, 10001)
        Reference lookahead: 1 time step
        -------------------
        Simulation mode: open_loop
        """

        print("Simulation settings");
        print("-------------------");
        print(f"Initial time step: {self._options["t0"]} s");
        print(f"Final time step: {self._options["tfinal"]} s");
        print(f"Solver step size: {self._solver.h} s");
        #print(f"Solver: {self._solver}");
        print("-------------------");
        print(f"Input signals format: {self._inputs.shape}");
        print(f"Output signals format: {self.outputs.shape}");
        print(f"Control actions format: {self.control_actions.shape}");
        print(f"Reference lookahead: {self._ref_lookahead} time step");
        print("-------------------");
        print(f"Simulation mode: {self._mode}");
        #print(f"Controller: {self.controller}");

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

        Examples
        --------
        >>> import numpy as np
        >>> from pynamics.models.state_space_models import LinearModel
        >>> from pynamics.simulations import Sim
        >>> 
        >>> A = np.array([[0, 0, -1], [1, 0, -3], [0, 1, -3]]);
        >>> B = np.array([1, -5, 1]).reshape(-1, 1);
        >>> C = np.array([0, 0, 1]);
        >>> D = np.array([0]);
        >>> model = LinearModel(np.zeros((3, 1)), np.zeros((1, 1)), A, B, C, D);
        >>>
        >>> simulation = Sim(model, input_signal=np.ones(int(10/0.001)+1));
        >>> res = simulation.run();
        >>> res
                 Time  Ref_1  u_1       y_1
        0       0.000    1.0  0.0  0.000000
        1       0.001    1.0  1.0  0.000996
        2       0.002    1.0  1.0  0.001984
        3       0.003    1.0  1.0  0.002964
        4       0.004    1.0  1.0  0.003936
        ...       ...    ...  ...       ...
        9996    9.996    1.0  1.0  0.984014
        9997    9.997    1.0  1.0  0.984026
        9998    9.998    1.0  1.0  0.984039
        9999    9.999    1.0  1.0  0.984052
        10000  10.000    1.0  1.0  0.984065
        <BLANKLINE>
        [10001 rows x 4 columns]
        """

        self.control_actions[:, 0] = self.system.get_input();
        self.outputs[:, 0] = self.system.get_output();

        #for ind, (t, _, y, n, _) in enumerate(zip(self.time[:-1], self.inputs[:, :-1], self.outputs[:, :-1], self.noise[:, :-1], self.control_actions[:, :-1])):
        for ind, t in enumerate(self.time[:-1]):

            #self.outputs[:, ind+1], self.control_actions[:, ind+1] = self._step(t, ref, y + n);
            self.outputs[:, ind+1], self.control_actions[:, ind+1] = self._step(t, self.inputs[:, ind:ind+self.ref_lookahead], self.outputs[:, ind:ind+1]); # + self.noise[:, ind:ind+1]
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

    def reset(self, initial_state: np.ndarray, initial_control: np.ndarray | float) -> None:
        """
        Reset simulation parameters (initial conditions, output arrays, control actions).

        This method must be called every time one wishes to run another simulation. The initial conditions, \
        output array and control actions array are all reset. This method is useful if one wishes to run \
        simulations with different initial conditions or different controllers.

        Parameters
        ----------
        initial_state : np.ndarray
            The system's initial state. Should be an array shaped (n, 1), where n \
            is the number of state variables.

        initial_control: np.ndarray
            The inputs' initial value(s). Should be an array shaped (u, 1), where
            u is the number of input variables.

        Examples
        --------
        >>> import numpy as np
        >>> from pynamics.models.state_space_models import LinearModel
        >>> from pynamics.simulations import Sim
        >>> 
        >>> A = np.array([[0, 0, -1], [1, 0, -3], [0, 1, -3]]);
        >>> B = np.array([1, -5, 1]).reshape(-1, 1);
        >>> C = np.array([0, 0, 1]);
        >>> D = np.array([0]);
        >>> model = LinearModel(np.zeros((3, 1)), np.zeros((1, 1)), A, B, C, D);
        >>>
        >>> simulation = Sim(model, input_signal=np.ones(int(10/0.001)+1));
        >>> res = simulation.run();
        >>> simulation.system.x
        array([[7.98056883],
               [1.96495125],
               [0.98406462]])
        >>>
        >>> simulation.reset(np.zeros((3, 1)), np.zeros((1, 1)));
        Sim outputs and control actions were reset sucessfully.
        >>> simulation.system.x
        array([[0.],
               [0.],
               [0.]])
        """

        self.system.x = initial_state;
        self.system.set_input(initial_control);
        self.outputs = np.zeros(shape=(self.system.output_dim, self.time.shape[0]));
        self.control_actions = np.zeros(shape=(self.controller.output_dim, self.time.shape[0]));
        print("Sim outputs and control actions were reset sucessfully.");
    
        return;

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
        Plot the reference signal and the system's output.

        Evaluate the system's tracking performance by plotting the reference signal and the system's output (SISO systems only).

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

        Examples
        --------
        >>> import numpy as np
        >>> from pynamics.models.state_space_models import LinearModel
        >>> from pynamics.simulations import Sim
        >>> 
        >>> A = np.array([[0, 0, -1], [1, 0, -3], [0, 1, -3]]);
        >>> B = np.array([1, -5, 1]).reshape(-1, 1);
        >>> C = np.array([0, 0, 1]);
        >>> D = np.array([0]);
        >>> model = LinearModel(np.zeros((3, 1)), np.zeros((1, 1)), A, B, C, D);
        >>>
        >>> simulation = Sim(model, input_signal=np.ones(int(10/0.001)+1));
        >>> res = simulation.run();
        >>> 
        >>> _ = Sim.tracking_plot(res, "Time", "Ref_1", "y_1");
        
        TODO : Add respective image!!!!
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
        Visualise the system's output signals.

        This method can be use to visualise the system's output signals simultaneously. \
        It supports MIMO systems.

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

        Examples
        --------
        >>> import numpy as np
        >>> from pynamics.models.state_space_models import LinearModel
        >>> from pynamics.simulations import Sim
        >>> 
        >>> A = np.array([[0, 0, -1], [1, 0, -3], [0, 1, -3]]);
        >>> B = np.array([1, -5, 1]).reshape(-1, 1);
        >>> C = np.array([0, 0, 1]);
        >>> D = np.array([0]);
        >>> model = LinearModel(np.zeros((3, 1)), np.zeros((1, 1)), A, B, C, D);
        >>>
        >>> simulation = Sim(model, input_signal=np.ones(int(10/0.001)+1));
        >>> res = simulation.run();
        >>> 
        >>> _ = Sim.system_outputs_plot(res, "Time", ["y_1"]);
        
        TODO : Add respective image!!!!
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

        This method can be used to simulate a system's step response. Keep in mind that, for now, it should only be used \
        with single-input systems or controllers needing only one reference signal.

        Parameters
        ----------
        system : BaseModel
            System to simulate. Must be described by a model supported by pynamics.
            TODO : Add link to 'pynamics' page.

        step_magnitude : int | float, default=1.0
            The step's magnitude. Unit step by default.

        t0 : float, default=0.0
            Initial time instant. Must be non-negative.

        tfinal : float, default=0.0
            Final time instant. Must be non-negative.

        solver : {"Euler", "Modified_Euler", "Heun", "RK4"}, str, default="RK4"
            Fixed-step solver.

        step_size : float, default=0.001
            Solver step size. Must be positive.

        mode : {"open_loop", "closed_loop"}, str, default="open_loop"
            Simulation mode. The controller will not be included in the simulation unless \
            parameter is set to "closed_loop".

        controller : BaseController | None, optional
            Controller.

        reference_labels : list[str] | None, optional
            List of labels for the reference signals.
                
        reference_lookahead : int, default=1
            Number of time steps ahead for which the reference values are known to the controller.
                
        noise_power : int | float, default=0.0
            White noise power. If equal to zero, no noise will be added to the simulation.

        noise_seed : int, default=0
            Random seed for the noise array.
        
        Returns
        -------
        Sim
            A simulation class instance.

        Examples
        --------
        >>> import numpy as np
        >>> from pynamics.models.state_space_models import LinearModel
        >>> from pynamics.simulations import Sim
        >>> 
        >>> A = np.array([[0, 0, -1], [1, 0, -3], [0, 1, -3]]);
        >>> B = np.array([1, -5, 1]).reshape(-1, 1);
        >>> C = np.array([0, 0, 1]);
        >>> D = np.array([0]);
        >>> model = LinearModel(np.zeros((3, 1)), np.zeros((1, 1)), A, B, C, D);
        >>>
        >>> simulation = Sim.step_response(model, step_magnitude=2);
        >>> res = simulation.run();
        >>> 
        >>> _ = Sim.tracking_plot(res, "Time", "Ref_1", "y_1");
        
        TODO : Add respective image!!!!
        """

        end = (tfinal - t0) / step_size + 1;
        reference_signal = np.full(shape=(1, int(end)), fill_value=step_magnitude);

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
        Simulate the system's response to a ramp signal.

        This method can be used to simulate a system's response to a ramp input. Keep in mind that, for now, it should only be used \
        with single-input systems or controllers needing only one reference signal.

        Parameters
        ----------
        system : BaseModel
            System to simulate. Must be described by a model supported by pynamics.
            TODO : Add link to 'pynamics' page.

        slope : int | float, default=1.0
            The ramp's slope. Unit ramp by default.

        t0 : float, default=0.0
            Initial time instant. Must be non-negative.

        tfinal : float, default=0.0
            Final time instant. Must be non-negative.

        solver : {"Euler", "Modified_Euler", "Heun", "RK4"}, str, default="RK4"
            Fixed-step solver.

        step_size : float, default=0.001
            Solver step size. Must be positive.

        mode : {"open_loop", "closed_loop"}, str, default="open_loop"
            Simulation mode. The controller will not be included in the simulation unless \
            parameter is set to "closed_loop".

        controller : BaseController | None, optional
            Controller.

        reference_labels : list[str] | None, optional
            List of labels for the reference signals.
                
        reference_lookahead : int, default=1
            Number of time steps ahead for which the reference values are known to the controller.
                
        noise_power : int | float, default=0.0
            White noise power. If equal to zero, no noise will be added to the simulation.

        noise_seed : int, default=0
            Random seed for the noise array.
        
        Returns
        -------
        Sim
            A simulation class instance.

        Warning
        -------
        This method seems to have some inaccuracies at the moment. Results might not be as reliable.

        Examples
        --------
        >>> import numpy as np
        >>> from pynamics.models.state_space_models import LinearModel
        >>> from pynamics.simulations import Sim
        >>> 
        >>> A = np.array([[0, 0, -1], [1, 0, -3], [0, 1, -3]]);
        >>> B = np.array([1, -5, 1]).reshape(-1, 1);
        >>> C = np.array([0, 0, 1]);
        >>> D = np.array([0]);
        >>> model = LinearModel(np.zeros((3, 1)), np.zeros((1, 1)), A, B, C, D);
        >>>
        >>> simulation = Sim.ramp(model, slope=1);
        >>> res = simulation.run();
        >>> 
        >>> _ = Sim.tracking_plot(res, "Time", "Ref_1", "y_1");
        
        TODO : Add respective image!!!!
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