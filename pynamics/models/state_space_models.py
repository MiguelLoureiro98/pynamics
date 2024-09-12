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

from .base import BaseModel
import numpy as np

"""
This module contains all state-space models supported by this package.

These include:
-> linearModel: a class for generic linear state-space models.
-> LTVModel: yet to be implemented.
-> LPVModel: yet to be implemented.
-> nonlinearModel: a class for generic nonlinear state-space models. 
                   Nonlinear time-varying systems can be defined using this class.
-> nonlinearPVModel: yet to be implemented.
"""

class LinearModel(BaseModel):

    """
    This class implements a generic linear state-space model. Its methods allow one to ... .

    Attributes
    ----------------------------------------------------------------------------------
    x: np.ndarray
    The system's state. Should be an array shaped (n, 1), where n is the number of
    variables.

    A: np.ndarray


    B: np.ndarray


    C: np.ndarray


    D: np.ndarray

    
    u: np.ndarray


    Methods
    ----------------------------------------------------------------------------------
    __init__
    _control_type_checks
    get_state
    get_output
    get_input
    set_input
    eval 
    """

    def __init__(self, initial_state: np.ndarray, initial_control: np.ndarray | float, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray,\
                 input_labels: list[str] | None=None, output_labels: list[str] | None=None) -> None:
        
        """
        Constructor method for the linearModel class.

        This method ... .

        Arguments
        ----------------------------------------------------------------------------------
        initial_state: np.ndarray
            The system's initial state. Should be an array shaped (n, 1), where
            n is the number of variables.

        initial_control: np.ndarray
            The inputs' initial value(s). Should be an array shaped (u, 1), where
            u is the number of input variables.

        A: np.ndarray


        B: np.ndarray


        C: np.ndarray


        D: np.ndarray

        
        u: np.ndarray


        Returns
        ----------------------------------------------------------------------------------
        None
        """

        self._matrix_type_checks(A, B, C, D);
        C, D = self._matrix_reformatting(C, D);
        super().__init__(initial_state, B.shape[1], C.shape[0], input_labels, output_labels);
        self.u = self._control_type_checks(initial_control);
        self._matrix_dim_checks(A, B, C, D);
        self.A = A;
        self.B = B;
        self.C = C;
        self.D = D;

        return;

    def _matrix_type_checks(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> None:

        """
        Helper method to ... .
        """

        test_A = isinstance(A, np.ndarray);
        test_B = isinstance(B, np.ndarray);
        test_C = isinstance(C, np.ndarray);
        test_D = isinstance(D, np.ndarray);

        if((test_A and test_B and test_C and test_D) is False):

            raise TypeError("Matrices A, B, C and D must be of np.ndarray type.");

        return;

    def _matrix_reformatting(self, C, D) -> tuple[np.ndarray]:

        """
        
        """

        if(len(C.shape) == 1):

            C = np.expand_dims(C, axis=0);
        
        if(len(D.shape) == 1):

            D = np.expand_dims(D, axis=1);
    
        return (C, D);

    def _matrix_dim_checks(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> None:

        """
        Helper method to ... .        
        """

        if(A.shape[0] != A.shape[1]):

            raise ValueError("A must be a square matrix.");

        if(A.shape[0] != B.shape[0] or C.shape[0] != D.shape[0] or B.shape[1] != D.shape[1] or A.shape[1] != C.shape[1]):

            raise ValueError("A must have the same number of rows as B, while C must have the same number of rows as D.\n \
                             Finally, B and D must have the same number of columns, and the same applies to A and C.");
    
        if(A.shape[0] != self.x.shape[0]):

            raise ValueError("A and B must have as many rows as x (the state vector) has columns.");

        if(B.shape[1] != self.u.shape[0]):

            raise ValueError("B must have as many columns as u (the input vector) has rows. The same must happen for D and u, respectively.");

        return;

    def info(self) -> None:

        """
        
        """

        return;

    def get_state(self) -> np.ndarray:
        
        """
        
        """

        return self.x;

    def get_output(self) -> np.ndarray:
        
        """
        
        """

        return np.matmul(self.C, self.x) + np.matmul(self.D, self.u);

    def get_input(self) -> np.ndarray:

        """
        
        """

        return self.u;

    def set_input(self, u: np.ndarray | float) -> None:

        """
        
        """

        self.u = self._control_type_checks(u);
    
        return;

    def update_state(self, state: np.ndarray) -> None:

        """
        
        """

        self.x = state;
    
        return;

    def eval(self, t: float, x: np.ndarray) -> np.ndarray:

        """
        
        """

        return np.matmul(self.A, x) + np.matmul(self.B, self.u);

class NonlinearModel(BaseModel):

    """
    This class implements a generic continuous nonlinear state-space model.
    Since both the state equations and the output equations are user-defined,
    time-varying systems are supported. In order to implement such a system,
    the state and/or the output equations should explicitly dependent on time.

    Attributes
    ----------------------------------------------------------------------------------
    x: np.ndarray
    The system's state. Should be an array shaped [of shape] (n, 1), where n is the number of
    variables.

    state_equations: callable
    The state equations. These describe the evolution of the system's state depending
    on its current state and its inputs.

    output_equations: callable
    The output equations. These establish the relations between the system's current
    state and its output. [These relate the system's state to its output.]

    u: np.ndarray
    The current control action, or set of control actions, defined as an (n, 1)-shaped
    array, where n is the number of controlled inputs. 

    Methods
    ----------------------------------------------------------------------------------
    __init__
    _control_type_checks
    get_state
    get_output
    get_input
    set_input
    eval 
    """

    def __init__(self, initial_state: np.ndarray, initial_control: np.ndarray, state_update_fcn: callable, state_output_fcn: callable, input_dim: int, output_dim: int,\
                 input_labels: list[str] | None=None, output_labels: list[str] | None=None) -> None:
        
        """
        Constructor method for the nonlinearModel class.

        This method ... .

        Arguments
        ----------------------------------------------------------------------------------
        initial_state: np.ndarray
        The system's initial state. Should be an array shaped (n, 1), where
        n is the number of variables.

        state_update_fcn: callable
        The state update function. If this class is used, it must be provided y the user.
        This function must receive ... .

        state_output_fcn: callable


        input_dim: int
        Number of inputs.

        output_dim: int
        Number of outputs.

        Returns
        ----------------------------------------------------------------------------------
        None
        """

        super().__init__(initial_state, input_dim, output_dim, input_labels, output_labels);
        self.state_equations = state_update_fcn;
        self.output_equations = state_output_fcn;
        self.u = self._control_type_checks(initial_control);

        return;

    def info(self) -> None:

        """
        
        """

        return;

    def get_state(self) -> np.ndarray:
        
        """
        
        """

        return self.x;

    def get_output(self) -> np.ndarray:
        
        """
        
        """

        return self.output_equations(self.x);

    def get_input(self) -> np.ndarray:

        """
        
        """

        return self.u;

    def set_input(self, u: np.ndarray | float) -> None:

        """
        
        """

        self.u = self._control_type_checks(u);
    
        return;

    def update_state(self, state: np.ndarray) -> None:

        """
        
        """

        self.x = state;
    
        return;

    def eval(self, t: float, x: np.ndarray) -> np.ndarray:

        """
        
        """

        return self.state_equations(x, self.u, t);

if __name__ == "__main__":

    pass