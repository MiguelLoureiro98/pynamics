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
This module contains the model base class, which forms the template for every plant model supported by this package.

Classes
-------
BaseModel
    Model base class.
"""

from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):

    """
    This is the parent class for every plant model supported by pynamics. 
    While custom models are supported, they must all inherit from this class.

    Parameters
    ----------
    initial_state : np.ndarray
        The system's initial state. Should be an array shaped (n, 1), where n is the number of \
        state variables.

    input_dim : int
        Number of system inputs.

    output_dim : int
        Number of system outputs.

    input_labels : list[str] | None, optional
        List of input labels. If `None` is passed, the inputs will be given generic names.

    output_labels : list[str] | None, optional
        List of output labels. If `None` is passed, the outputs will be given generic names.

    Attributes
    ----------
    x : np.ndarray
        The system's state vector.

    input_dim : int
        Number of system inputs.

    output_dim : int
        Number of system outputs.

    state_dim : int
        Number of states in the state vector.

    input_labels : list[str]
        List of input labels.

    output_labels : list[str]
        List of output labels.

    Methods
    -------
    get_state()
        Get the current state vector.

    get_output()
        Compute the system's output from the current state.

    get_input()
        Get the current inputs.

    set_input(u: np.ndarray | float)
        Pass new inputs to the system.
        
    update_state()
        Assign new values to the system's state vector.

    eval()
        Compute the system's state derivative.

    Warning
    -------
    This is an abstract base class. It should not be used directly. 
    """

    def __init__(self, 
                 initial_state: np.ndarray, 
                 input_dim: int, 
                 output_dim: int,
                 input_labels: list[str] | None=None, 
                 output_labels: list[str] | None=None) -> None:
        
        """
        Class constructor.
        """

        super().__init__();
        self.x = initial_state;
        self._dim_checks(input_dim, output_dim);
        self.input_dim = input_dim;
        self.output_dim = output_dim;
        self.state_dim = self.x.shape[0];
        self.input_labels = self._labels_check(input_labels, self.input_dim, "u");
        self.output_labels = self._labels_check(output_labels, self.output_dim, "y");

        return;

    def _control_type_checks(self, control_action: np.ndarray | float | int) -> np.ndarray:
        """
        Check and reformat control actions.
        """

        if (isinstance(control_action, float) is True or isinstance(control_action, int) is True):

            control_action = np.array([control_action]);
        
        if(control_action.shape[0] == 1 and len(control_action.shape) == 1):

            control_action = np.expand_dims(control_action, axis=1);

        return control_action;

    def _dim_checks(self, input_dim: int, output_dim: int) -> None:
        """
        Check input and output dimensions.
        """

        if((isinstance(input_dim, int) and isinstance(output_dim, int)) is False):

            raise TypeError("Both the input and output dimensions should be integers.");

        return;

    def _labels_check(self, labels: list[str], dim: int, char: str) -> list[str]:
        """
        Perform label and type checks on the labels. Reformat them if necessary.
        """

        if(labels is None):

            new_labels = [f"{char}_{num}" for num in range(1, dim + 1)];
        
        else:

            if(isinstance(labels, list) is False):

                raise TypeError("Both 'input_labels' and 'output_labels' must be lists.");
    
            elif(len(labels) != dim):

                raise ValueError("The number of labels does not match the dimensions.");
    
            new_labels = labels;

        return new_labels;
    
    @abstractmethod
    def get_state(self) -> np.ndarray:
        """
        Access the system's state.

        Abstract method. Implementation details may vary with the model.
        """

        pass

    @abstractmethod
    def get_output(self) -> np.ndarray:
        """
        Compute the system's output from the current state vector.

        Abstract method. Implementation details may vary with the model.
        """

        pass

    @abstractmethod
    def get_input(self) -> np.ndarray:
        """
        Access the system's input.

        Abstract method. Implementation details may vary with the model.
        """

        pass

    @abstractmethod
    def set_input(self, u: np.ndarray | float) -> None:
        """
        Pass a new set of inputs (references, control actions, etc.) to the system.

        Abstract method. Implementation details may vary with the model.
        """

        pass

    @abstractmethod
    def update_state(self) -> None:
        """
        Assign new values to the system's state vector.

        Abstract method. Implementation details may vary with the model.
        """

        pass

    @abstractmethod
    def eval(self) -> np.ndarray:
        """
        Compute the system's state derivative.

        Abstract method. Implementation details may vary with the model.
        """

        pass