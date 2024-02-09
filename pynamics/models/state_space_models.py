from ._model import model
import numpy as np

class linearModel(model):

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

    def __init__(self, initial_state: np.ndarray, initial_control: np.ndarray | float, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> None:
        
        """
        Constructor method for the linearModel class.

        This method ... .

        Arguments
        ----------------------------------------------------------------------------------
        initial_state: np.ndarray
        The system's initial state. Should be an array shaped (n, 1), where
        n is the number of variables.

        A: np.ndarray


        B: np.ndarray


        C: np.ndarray


        D: np.ndarray

        
        u: np.ndarray


        Returns
        ----------------------------------------------------------------------------------
        None
        """

        super().__init__(initial_state)
        self.A = A;
        self.B = B;
        self.C = C;
        self.D = D;
        self.u = self._control_type_checks(initial_control);

        return;

    def _matrix_checks(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> None:

        test_A = isinstance(A, np.ndarray);
        test_B = isinstance(B, np.ndarray);
        test_C = isinstance(C, np.ndarray);
        test_D = isinstance(D, np.ndarray);

        if((test_A and test_B and test_C and test_D) is False):

            raise TypeError("Matrices A, B, C and D must be of np.ndarray type.");
    
        # Further testing: matrix dimensions -> check appropriate exception (value error?)

        return;

    def get_state(self) -> np.ndarray:
        
        """
        
        """

        return self.x;

    def get_output(self) -> np.ndarray:
        
        """
        
        """

        return np.matmul(self.C, self.x);

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

class nonlinearModel(model):

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

    def __init__(self, initial_state: np.ndarray, initial_control: np.ndarray, state_update_fcn: callable, state_output_fcn: callable) -> None:
        
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


        Returns
        ----------------------------------------------------------------------------------
        None
        """

        super().__init__(initial_state);
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