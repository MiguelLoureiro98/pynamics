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

import numpy as np

"""
This module contains a dummy controller useful for open-loop simulations.
"""

class dummy_controller(object):

    """
    This class defines the dummy controller used by the Pynamics package to run
    open-loop simulations.
    The controller performs no computations.

    Attributes
    ----------------------------------------------------------------------------------
    input_dim: int
    The number of controller inputs (it should be one for a single-input controller).

    output_dim: int
    The number of controller outputs (it should be one for a single-output controller).

    Methods
    ----------------------------------------------------------------------------------
    __init__
    info
    control
    """

    def __init__(self, n_inputs: int, n_outputs: int, sampling_time: int | float) -> None:

        """
        
        """
        
        self.Ts = sampling_time;
        self.input_dim = n_inputs;
        self.output_dim = n_outputs;
        
        return;

    def info(self) -> None:

        """
        
        """

        print("-----------------------------------------------------------------");
        print("Pynamics Dummy Controller");
        print("-----------------------------------------------------------------");
        print("Description:");
        print("Pynamics makes use of this class in open-loop simulations.\
              It is not really a controller, as its output will simply be\
              the reference signal. No computations are performed.");
        print("-----------------------------------------------------------------");
        print("WARNING: for the reasons stated above, this controller should NOT be used\
              as a baseline. Its use is equivalent to an open-loop simulation.");
        print("-----------------------------------------------------------------");

        return;

    def control(self, ref: np.ndarray, y: np.ndarray) -> np.ndarray:

        """
        
        """

        return ref;