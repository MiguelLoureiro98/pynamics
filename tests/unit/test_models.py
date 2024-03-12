import unittest
import numpy as np
from pynamics.models.state_space_models import linearModel, nonlinearModel

"""

"""

def state_function_test(state: np.ndarray, control: np.ndarray, time: float) -> np.ndarray:

    """
    
    """

    state_derivative_1 = state[0] + 2 * np.cos(state[1]);
    state_derivative_2 = state[1] + control[0];

    state_derivative = np.array([state_derivative_1, state_derivative_2]);
    
    return state_derivative;

def MIMO_state_function_test(state: np.ndarray, control: np.ndarray, time: float) -> np.ndarray:

    """
    
    """

    state_derivative_1 = state[0] + 2 * np.cos(state[1]) - control[1];
    state_derivative_2 = state[1] + control[0];

    state_derivative = np.array([state_derivative_1, state_derivative_2]);
    
    return state_derivative;

def time_dependent_state_function_test(state: np.ndarray, control: np.ndarray, time: float) -> np.ndarray:

    """
    
    """

    state_derivative_1 = state[0] * time + 2 * np.cos(state[1]);
    state_derivative_2 = state[1] + control[0] + 3 * time;

    state_derivative = np.array([state_derivative_1, state_derivative_2]);
    
    return state_derivative;

def MIMO_time_dependent_state_function_test(state: np.ndarray, control: np.ndarray, time: float) -> np.ndarray:

    """
    
    """

    state_derivative_1 = state[0] * time + 2 * np.cos(state[1]) - control[1];
    state_derivative_2 = state[1] + control[0] + 3 * time;

    state_derivative = np.array([state_derivative_1, state_derivative_2]);
    
    return state_derivative;

def output_function_test(state: np.ndarray) -> np.ndarray:

    """
    
    """

    output = np.array([state[0]**state[1]]);

    return output;

def MIMO_output_function_test(state: np.ndarray) -> np.ndarray:

    """
    
    """

    output = np.array([[state[0]**state[1]], [state[0] + state[2]]]);

    return output;

class TestModels(unittest.TestCase):

    """
    
    """

    @classmethod
    def setUpClass(cls) -> None:

        """
        Unused.
        """

        pass

    @classmethod
    def tearDownClass(cls) -> None:
        
        """
        Unused.
        """

        pass

    def setUp(self) -> None:

        A = np.array([[0, 0, -1], [1, 0, -3], [0, 1, -3]]);
        B = np.array([1, -5, 1]).reshape(-1, 1);
        C = np.array([0, 0, 1]);
        D = np.array([0]);
        linmodel = linearModel(np.zeros((3, 1)), np.array([0]), A, B, C, D);

        nlinmodel = nonlinearModel(np.zeros((3, 1)), np.array([0]), state_function_test, output_function_test, 1, 1);
        time_variant_model = nonlinearModel(np.zeros((3, 1)), np.array([0]), time_dependent_state_function_test, output_function_test, 1, 1);

        self.linear_model = linmodel;
        self.nonlinear_model = nlinmodel;
        self.time_varying_model = time_variant_model;

    def tearDown(self) -> None:
        
        del self.linear_model
        print("Deleted linear model.");
    
        del self.nonlinear_model;
        print("Deleted nonlinear model.");
    
        del self.time_varying_model;
        print("Deleted time-varying model.");

    def test_initialisation(self) -> None:

        pass

    def test_getters(self) -> None:

        """
        
        """

        # Test MIMO models as well!!!

        self.assertEqual(self.linear_model.get_input(), np.array([0]));
        self.assertEqual(self.nonlinear_model.get_input(), np.array([0]));
        self.assertEqual(self.time_varying_model.get_input(), np.array([0]));
    
        self.assertEqual(self.linear_model.get_state()[0], np.array([0]));
        self.assertEqual(self.linear_model.get_state()[1], np.array([0]));
        self.assertEqual(self.linear_model.get_state()[2], np.array([0]));
        self.assertEqual(self.nonlinear_model.get_state()[0], np.array([0]));
        self.assertEqual(self.nonlinear_model.get_state()[1], np.array([0]));
        self.assertEqual(self.nonlinear_model.get_state()[2], np.array([0]));
        self.assertEqual(self.time_varying_model.get_state()[0], np.array([0]));
        self.assertEqual(self.time_varying_model.get_state()[1], np.array([0]));
        self.assertEqual(self.time_varying_model.get_state()[2], np.array([0]));
    
        self.assertEqual(self.linear_model.get_output(), np.array([0]));
        self.assertEqual(self.nonlinear_model.get_output(), np.array([1]));
        self.assertEqual(self.time_varying_model.get_output(), np.array([1]));

    def test_setters(self) -> None:

        pass

    def test_update(self) -> None:

        pass

    def test_eval(self) -> None:

        pass

if __name__ == "__main__":
    unittest.main();