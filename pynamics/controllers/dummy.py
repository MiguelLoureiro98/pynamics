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

from .base import BaseController
import numpy as np

"""
This module contains a dummy controller useful for open-loop simulations.

Classes
-------
DummyController
    A dummy controller that returns the reference value(s). Used to run open-loop simulations.
"""

class DummyController(BaseController):
    """
    This class defines the dummy controller used by the pynamics package to run
    open-loop simulations.
    The controller performs no computations.

    Attributes
    ----------
    input_dim: int
        The number of controller inputs.

    output_dim: int
        The number of control actions (one for a single-output controller / single-input system).

    Ts : int | float
        Controller sampling time.

    Methods
    -------
    info()
        Display a warning.

    control(ref: np.ndarray, y: np.ndarray)
        Compute the control actions for the next time instant. In practice, it simply outputs the value of `ref`.
    """

    def __init__(self, n_inputs: int, n_outputs: int, sampling_time: int | float) -> None:
        """
        Class constructor.
        """
        
        super().__init__(n_inputs, n_outputs, sampling_time);
        
        return;

    def info(self) -> None:
        """
        Provides useful information regarding the controller.

        This method issues a warning to the user that this 'controller' should not be used \
        as a benchmark of any kind. It is simply used for open-loop simulations.
        """

        print("pynamics Dummy Controller");
        print("-------------------------");
        print("pynamics makes use of this class in open-loop simulations. \
              It is not really a controller, as its output will simply be \
              the reference signal. No computations are performed.");
        print("WARNING: for the reasons stated above, this controller should NOT be used \
              as a baseline. Its use is equivalent to an open-loop simulation.");

        return;

    def control(self, ref: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes the control actions for the next time instant.

        This method outputs the control actions for the next time instant, which in this case are simply the reference values.

        Parameters
        ----------
        ref : np.ndarray
            Array of reference values. For regulation problems, this should be an array of zeros.

        y : np.ndarray
            System state vector. Used for compatibility reasons. Unused by this 'controller'.

        Returns
        -------
        np.ndarray
            Array of control actions. In this case, the reference values themselves.
        """

        return ref;