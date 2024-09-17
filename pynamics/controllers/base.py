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
This module contains the controller base class. Every controller class compatible with pynamics should inherit from this class.

Classes
-------
BaseController
    Controller base class.
"""

from abc import ABC, abstractmethod
import numpy as np

class BaseController(ABC):
    """
    Base class for all controllers supported by this package.

    Every controller class compatible with pynamics should inherit from this class. This is simply an abstract base class.
    As such, it should not be used directly. Derived classes may naturally provide methods and attributes that are not 
    strictly required by this base class.

    Parameters
    ----------
    n_inputs : int
        Number of controller inputs.

    n_outputs : int
        Number of controller outputs.\
        Should be the same as the system's input dimension.

    sampling_time : int | float
        Controller sampling time (in seconds).

    Attributes
    ----------
    input_dim : int
        Number of controller inputs.

    output_dim : int
        Number of controller outputs.

    Ts : int | float
        Controller sampling time (in seconds).

    Methods
    -------
    info()
        Display useful information regarding the controller.

    control(ref: np.ndarray, y: np.ndarray)
        Compute control actions for the next time instant.

    Warning
    -------
    This is an abstract base class. It should not be used directly. 
    """

    def __init__(self, n_inputs: int, n_outputs: int, sampling_time: int | float) -> None:

        """
        Class constructor.
        """

        super().__init__();
        self._input_dim = n_inputs;
        self._output_dim = n_outputs;
        self._sampling_time = sampling_time;

        return;

    @property
    def input_dim(self) -> int:
        """
        Get the controller's input dimension.

        This method can be used to access the number of controller input variables using dot notation.

        Returns
        -------
        int
            Number of controller inputs.
        """

        return self._input_dim;

    @property
    def output_dim(self) -> int:
        """
        Get the controller's output dimension.

        This method can be used to access the number of controller output variables using dot notation.

        Returns
        -------
        int
            Number of controller outputs.
        """

        return self._output_dim;

    @property
    def Ts(self) -> int | float:
        """
        Get the controller's sampling time.

        This method can be used to access the controller's sampling time using dot notation.

        Returns
        -------
        int | float
            Controller sampling time (seconds).
        """

        return self._sampling_time;

    @Ts.setter
    def Ts(self, new_sampling_time: int | float) -> None:
        """
        Set the controller's sampling time.

        This method can be used to modify a controller's sampling time using dot notation.

        Parameters
        ----------
        new_sampling_time : int | float
            New sampling time.
        """

        self._sampling_time = new_sampling_time;
    
        return;

    @abstractmethod
    def info(self) -> None:
        """
        Provides useful information regarding the controller.

        Abstract method. Implementation details are specific to each controller and should therefore vary considerably.
        """

        pass

    @abstractmethod
    def control(self, ref: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes the control actions for the next time instant.

        Abstract method. Implementation details are specific to each controller and should therefore vary considerably.

        Parameters
        ----------
        ref : np.ndarray
            Array of reference values. For regulation problems, this should be an array of zeros.

        y : np.ndarray
            System state vector. Note that this refers only to the latest time instant. \
            If a controller requires information from earlier time instants, it should store \
            it internally.

        Returns
        -------
        np.ndarray
            Array of control actions.
        """

        pass