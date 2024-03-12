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

    def __init__(self, n_inputs: int, n_outputs: int) -> None:

        """
        
        """
        
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