from abc import ABC, abstractmethod
import numpy as np

"""
This module contains the model base class, which forms the template for every plant model supported by this package.
"""

class model(ABC):

    """
    This is the parent class for every plant model supported by Pynamics. 
    While custom models are supported, they must all inherit from this class.

    Attributes
    ----------------------------------------------------------------------------------
    x: np.ndarray
    The system's state. Should be an array shaped (n, 1), where n is the number of
    state variables.

    Methods
    ----------------------------------------------------------------------------------
    __init__
    get_state
    get_output
    get_input
    set_input
    eval 
    """

    def __init__(self, initial_state: np.ndarray, input_dim: int, output_dim: int, \
                 input_labels: list | None=None, output_labels: list | None=None, state_labels: list | None=None) -> None:
        
        """
        Class constructor. Receives the system's initial state as an input.

        Arguments
        ----------------------------------------------------------------------------------
        initial_state: np.ndarray
        The system's initial state. Should be an array shaped (n, 1), where
        n is the number of variables.

        input_dim: int
        Number of inputs.

        output_dim: int
        Number of outputs.

        input_labels: list or None


        output_labels: list or None

        
        Returns
        ----------------------------------------------------------------------------------
        None
        """

        super().__init__();
        self.x = initial_state;
        self.input_dim = input_dim;
        self.output_dim = output_dim;
        self.state_dim = self.x.shape[0];

        if(input_labels is None):

            input_labels = [f"Input_{num}" for num in range(1, self.input_dim + 1)];
        
        if(output_labels is None):

            output_labels = [f"Input_{num}" for num in range(1, self.output_dim + 1)];
        
        if(state_labels is None):

            state_labels = [f"Input_{num}" for num in range(1, self.state_dim + 1)];
        
        self.input_labels = input_labels;
        self.output_labels = output_labels;
        self.state_labels = state_labels;

        return;

    def _control_type_checks(self, control_action: np.ndarray | float) -> np.ndarray:

        """
        Internal helper method to perform the necessary checks when a new control action
        is defined.

        Arguments
        ----------------------------------------------------------------------------------
        control_action: np.ndarray | float
        The new control action. It can be specified as a float, a flat array, or a ...

        Returns
        ----------------------------------------------------------------------------------
        control_action: np.ndarray
        The same control action ... [in the right format].
        """

        if (isinstance(control_action, float) is True):

            control_action = np.array([control_action]);
        
        if(control_action.shape[0] == 1):

            control_action = np.expand_dims(control_action, axis=1);

        return control_action;

    @abstractmethod
    def info(self) -> None:

        """
        Method to provid general information regarding model structure, parameters, etc.
        """

        pass
    
    @abstractmethod
    def get_state(self) -> np.ndarray:

        """
        Method to access the system's state.
        """

        pass

    @abstractmethod
    def get_output(self) -> np.ndarray:

        """
        Method to access the system's output.
        """

        pass

    @abstractmethod
    def get_input(self) -> np.ndarray:

        """
        Method to access the system's input.
        """

        pass

    @abstractmethod
    def set_input(self, u: np.ndarray | float) -> None:

        """
        Method to set a new set of inputs (references, control actions, etc.).
        """

        pass

    @abstractmethod
    def update_state(self) -> None:

        """
        Method to update the system's state.
        """

        pass

    @abstractmethod
    def eval(self) -> np.ndarray:

        """
        Method used to compute the model's state derivative at a given time instant.
        """

        pass