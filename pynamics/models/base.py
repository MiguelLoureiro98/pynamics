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

from abc import ABC, abstractmethod
import numpy as np

"""
This module contains the model base class, which forms the template for every plant model supported by this package.
"""

class BaseModel(ABC):

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
                 input_labels: list[str] | None=None, output_labels: list[str] | None=None) -> None:
        
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
        self._dim_checks(input_dim, output_dim);
        self.input_dim = input_dim;
        self.output_dim = output_dim;
        self.state_dim = self.x.shape[0];
        self.input_labels = self._labels_check(input_labels, self.input_dim, "u");
        self.output_labels = self._labels_check(output_labels, self.output_dim, "y");
        #self.state_labels = state_labels;

        return;

    def _control_type_checks(self, control_action: np.ndarray | float | int) -> np.ndarray:

        """
        Internal helper method to perform the necessary checks when a new control action
        is defined.

        Arguments
        ----------------------------------------------------------------------------------
        control_action: np.ndarray | float | int
        The new control action. It can be specified as a float, a flat array, or a ...

        Returns
        ----------------------------------------------------------------------------------
        control_action: np.ndarray
        The same control action ... [in the right format].
        """

        if (isinstance(control_action, float) is True or isinstance(control_action, int) is True):

            control_action = np.array([control_action]);
        
        if(control_action.shape[0] == 1 and len(control_action.shape) == 1):

            control_action = np.expand_dims(control_action, axis=1);

        return control_action;

    def _dim_checks(self, input_dim: int, output_dim: int) -> None:

        """
        
        """

        if((isinstance(input_dim, int) and isinstance(output_dim, int)) is False):

            raise TypeError("Both the input and output dimensions should be integers.");

        return;

    def _labels_check(self, labels: list[str], dim: int, char: str) -> list[str]:

        """
        
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