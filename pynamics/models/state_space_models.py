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

Classes
-------
LinearModel
    Implements a linear time-invariant state-space model.

NonlinearModel
    Implements a generic nonlinear model. Supports time-varying and parameter-varying models (including linear ones).
"""

class LinearModel(BaseModel):
    """
    Linear time-invariant state-space model.

    This class implements a generic linear time-invariant state-space model. \
    It is intended to model continuous-time systems. Hybrid and discrete-time \
    systems are not supported by pynamics.

    Parameters
    ----------
    initial_state : np.ndarray
        The system's initial state. Should be an array shaped (n, 1), where \
        n is the number of state variables.

    initial_control : np.ndarray
        The inputs' initial value(s). Should be an array shaped (u, 1), where \
        u is the number of input variables.

    A : np.ndarray
        Dynamics matrix. It maps the state vector to the state derivatives.

    B : np.ndarray
        Input matrix. It maps the input vector to the state derivatives.

    C : np.ndarray
        Output matrix. It maps the state vector to the output vector.

    D : np.ndarray
        Direct or feedforward matrix. It maps the input vector to the output vector.

    input_labels : list[str] | None, optional
        List of input labels. If `None` is passed, the inputs will be given generic names.

    output_labels : list[str] | None, optional
        List of output labels. If `None` is passed, the outputs will be given generic names.

    Attributes
    ----------
    x : np.ndarray
        The system's state vector. Should be an array shaped (n, 1), where n is the \
        number of state variables.

    u : np.ndarray
        Input vector. Should be an array shaped (u, 1), where \
        u is the number of input variables.

    A : np.ndarray
        Dynamics matrix. It maps the state vector to the state derivatives.

    B : np.ndarray
        Input matrix. It maps the input vector to the state derivatives.

    C : np.ndarray
        Output matrix. It maps the state vector to the output vector.

    D : np.ndarray
        Direct or feedforward matrix. It maps the input vector to the output vector.

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

    Raises
    ------
    TypeError
        If any of the matrices is not an np.array.

    ValueError
        If any of the matrices has incorrect dimensions.

    See also
    --------
    NonlinearModel:
        Implements a nonlinear state-space model. Supports linear time-varying systems and linear parameter-varying systems, \
        as well as generic nonlinear systems.
    TODO : Add page link.
    
    Notes
    -----
    A linear state-space model describes the relation between the inputs and outputs of a system as a set \
    of linear, first-order differential equations. These equations may be written in matrix form:

    $$
    \dot{x}(t) = A \cdot x(t) + B \cdot u(t)
    y(t) = C \cdot x(t) + D \cdot u(t)
    $$

    where $x$ is called the state vector, $u$ is the input vector and $y$ is the output vector. 
    The first equation is called the state equation, and the second one is the output equation. 
    The former relates the system's current state and the inputs to the change in state, while the latter \
    relates those same quantities to the system's outputs.

    If the system is time-invariant (as assumed by this class), then the A, B, C and D matrices are constant.
    """

    def __init__(self, initial_state: np.ndarray, 
                 initial_control: np.ndarray | float, 
                 A: np.ndarray, 
                 B: np.ndarray, 
                 C: np.ndarray, 
                 D: np.ndarray,
                 input_labels: list[str] | None=None, 
                 output_labels: list[str] | None=None) -> None:
        
        """
        Class constructor.
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
        Matrix type checks.
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
        Reformat C and D matrices if necessary.
        """

        if(len(C.shape) == 1):

            C = np.expand_dims(C, axis=0);
        
        if(len(D.shape) == 1):

            D = np.expand_dims(D, axis=1);
    
        return (C, D);

    def _matrix_dim_checks(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> None:
        """
        Check matrix dimensions.        
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

    def get_state(self) -> np.ndarray:  
        """
        Access the system's state.

        This method allows one to access the current state vector.

        Returns
        -------
        np.ndarray
            Current state vector.

        Examples
        --------
        >>> import numpy as np
        >>> from pynamics.models.state_space_models import LinearModel
        >>>
        >>> A = np.array([[0, 0, -1], [1, 0, -3], [0, 1, -3]]);
        >>> B = np.array([1, -5, 1]).reshape(-1, 1);
        >>> C = np.array([0, 0, 1]);
        >>> D = np.array([0]);
        >>> model = LinearModel(np.zeros((3, 1)), np.zeros((1, 1)), A, B, C, D);
        >>>
        >>> model.get_state()
        array([[0.],
               [0.],
               [0.]]) 
        """

        return self.x;

    def get_output(self) -> np.ndarray:
        """
        Compute the system's output from the current state vector.

        This method can be used to compute the output of a linear state-space \
        model from its current state vector. This is done by computing \
        $y = C \cdot x + D \cdot u$, where $y$ is the output vector, $x$ is the state vector, \
        and $u$ is the input vector.

        Returns
        -------
        np.ndarray
            Output vector.

        Examples
        --------
        >>> import numpy as np
        >>> from pynamics.models.state_space_models import LinearModel
        >>>
        >>> A = np.array([[0, 0, -1], [1, 0, -3], [0, 1, -3]]);
        >>> B = np.array([1, -5, 1]).reshape(-1, 1);
        >>> C = np.array([0, 0, 1]);
        >>> D = np.array([0]);
        >>> model = LinearModel(np.ones((3, 1)), np.zeros((1, 1)), A, B, C, D);
        >>>
        >>> model.get_output()
        array([[1.]])
        """

        return np.matmul(self.C, self.x) + np.matmul(self.D, self.u);

    def get_input(self) -> np.ndarray:
        """
        Access the system's input.

        This method can be use to access the current input vector.

        Returns
        -------
        np.ndarray
            Current input vector.

        Examples
        --------
        >>> import numpy as np
        >>> from pynamics.models.state_space_models import LinearModel
        >>>
        >>> A = np.array([[0, 0, -1], [1, 0, -3], [0, 1, -3]]);
        >>> B = np.array([1, -5, 1]).reshape(-1, 1);
        >>> C = np.array([0, 0, 1]);
        >>> D = np.array([0]);
        >>> model = LinearModel(np.zeros((3, 1)), np.ones((1, 1)), A, B, C, D);
        >>>
        >>> model.get_input()
        array([[1.]])
        """

        return self.u;

    def set_input(self, u: np.ndarray | float) -> None:
        """
        Pass a new set of inputs (references, control actions, etc.) to the system.

        This method can be used to update the system's input vector directly.

        Parameters
        ----------
        u : np.ndarray | float
            The new set of inputs (i.e. input vector).

        Examples
        --------
        >>> import numpy as np
        >>> from pynamics.models.state_space_models import LinearModel
        >>>
        >>> A = np.array([[0, 0, -1], [1, 0, -3], [0, 1, -3]]);
        >>> B = np.array([1, -5, 1]).reshape(-1, 1);
        >>> C = np.array([0, 0, 1]);
        >>> D = np.array([0]);
        >>> model = LinearModel(np.zeros((3, 1)), np.ones((1, 1)), A, B, C, D);
        >>>
        >>> model.get_input()
        array([[1.]])
        >>>
        >>> model.set_input(np.array([[5]]));
        >>> model.get_input()
        array([[5.]])

        For single-input systems, floats and integers also constitute valid inputs:

        >>> model.set_input(2.5);
        >>> model.get_input()
        array([[2.5]])
        """

        self.u = self._control_type_checks(u);
    
        return;

    def update_state(self, state: np.ndarray) -> None:
        """
        Assign new values to the system's state vector.

        This method can be used to update the system's state vector directly.

        Parameters
        ----------
        state : np.ndarray
            New state vector.

        Examples
        --------
        >>> import numpy as np
        >>> from pynamics.models.state_space_models import LinearModel
        >>>
        >>> A = np.array([[0, 0, -1], [1, 0, -3], [0, 1, -3]]);
        >>> B = np.array([1, -5, 1]).reshape(-1, 1);
        >>> C = np.array([0, 0, 1]);
        >>> D = np.array([0]);
        >>> model = LinearModel(np.zeros((3, 1)), np.zeros((1, 1)), A, B, C, D);
        >>> 
        >>> model.get_state()
        array([[0.],
               [0.],
               [0.]])
        >>>
        >>> model.update_state(np.ones((3, 1)));
        >>> model.get_state()
        array([[1.],
               [1.],
               [1.]])
        """

        self.x = state;
    
        return;

    def eval(self, t: float, x: np.ndarray) -> np.ndarray:
        """
        Compute the system's state derivative.

        This method computes the system's state derivative via the state \
        equation: $\dot{x} = A \cdot x + B \cdot u$, where $x$ is the state vector \
        and $u$ is the input vector.

        Parameters
        ----------
        t : float
            Time instant. Used for compatibility reasons. Unused by this method.

        x : np.ndarray
            The current state vector.

        Returns
        -------
        np.ndarray
            The system's state derivative.

        Examples
        --------
        >>> import numpy as np
        >>> from pynamics.models.state_space_models import LinearModel
        >>>
        >>> A = np.array([[0, 0, -1], [1, 0, -3], [0, 1, -3]]);
        >>> B = np.array([1, -5, 1]).reshape(-1, 1);
        >>> C = np.array([0, 0, 1]);
        >>> D = np.array([0]);
        >>> model = LinearModel(np.ones((3, 1)), np.zeros((1, 1)), A, B, C, D);
        >>>
        >>> model.eval(0.0, x=model.get_state())
        array([[-1.],
               [-2.],
               [-2.]])
        """

        return np.matmul(self.A, x) + np.matmul(self.B, self.u);

class NonlinearModel(BaseModel):

    """
    Generic nonlinear state-space model.

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