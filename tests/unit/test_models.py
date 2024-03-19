import unittest
import numpy as np
from pynamics.models.state_space_models import linearModel, nonlinearModel

"""
Test cases to test linear and nonlinear state-space models.
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
    state_derivative_3 = 1 - 2 * state[2];

    state_derivative = np.array([state_derivative_1, state_derivative_2, state_derivative_3]);
    
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
    state_derivative_3 = 1 - 2 * state[2];

    state_derivative = np.array([state_derivative_1, state_derivative_2, state_derivative_3]);
    
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

        # SISO models

        A = np.array([[0, 0, -1], [1, 0, -3], [0, 1, -3]]);
        B = np.array([1, -5, 1]).reshape(-1, 1);
        C = np.array([0, 0, 1]);
        D = np.array([0]);
        linmodel = linearModel(np.zeros((3, 1)), np.array([0]), A, B, C, D);

        nlinmodel = nonlinearModel(np.zeros((2, 1)), np.array([0]), state_function_test, output_function_test, 1, 1);
        time_variant_model = nonlinearModel(np.zeros((2, 1)), np.array([0]), time_dependent_state_function_test, output_function_test, 1, 1);

        self.linear_model = linmodel;
        self.nonlinear_model = nlinmodel;
        self.time_varying_model = time_variant_model;
    
        # MIMO models
    
        Bm = np.array([[1, 1], [-5, 0], [1, 0]]);
        Cm = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
        Dm = np.array([[0, 0], [0, 0], [0, 1]]);
        MIMO_linmodel = linearModel(np.zeros((3, 1)), np.array([[0], [0]]), A, Bm, Cm, Dm);
    
        MIMO_nlinmodel = nonlinearModel(np.zeros((3, 1)), np.array([[0], [0]]), MIMO_state_function_test, MIMO_output_function_test, 2, 2);
        MIMO_time_variant_model = nonlinearModel(np.zeros((3, 1)), np.array([[0], [0]]), MIMO_time_dependent_state_function_test, MIMO_output_function_test, 2, 2);
    
        self.MIMO_linear = MIMO_linmodel;
        self.MIMO_nonlinear = MIMO_nlinmodel;
        self.MIMO_time_varying = MIMO_time_variant_model;

    def tearDown(self) -> None:
        
        del self.linear_model
        print("Deleted linear model.");
    
        del self.nonlinear_model;
        print("Deleted nonlinear model.");
    
        del self.time_varying_model;
        print("Deleted time-varying model.");

    def test_initialisation(self) -> None:

        """
        
        """

        pass

    def test_getters(self) -> None:

        """
        
        """

        # SISO models

        self.assertEqual(self.linear_model.get_input(), np.array([0]));
        self.assertEqual(self.nonlinear_model.get_input(), np.array([0]));
        self.assertEqual(self.time_varying_model.get_input(), np.array([0]));
    
        self.assertEqual(self.linear_model.get_state()[0], np.array([0]));
        self.assertEqual(self.linear_model.get_state()[1], np.array([0]));
        self.assertEqual(self.linear_model.get_state()[2], np.array([0]));
        self.assertEqual(self.nonlinear_model.get_state()[0], np.array([0]));
        self.assertEqual(self.nonlinear_model.get_state()[1], np.array([0]));
        self.assertEqual(self.time_varying_model.get_state()[0], np.array([0]));
        self.assertEqual(self.time_varying_model.get_state()[1], np.array([0]));
    
        self.assertEqual(self.linear_model.get_output(), np.array([0]));
        self.assertEqual(self.nonlinear_model.get_output(), np.array([1]));
        self.assertEqual(self.time_varying_model.get_output(), np.array([1]));
    
        # MIMO models
    
        self.assertEqual(self.MIMO_linear.get_input()[0], np.array([0]));
        self.assertEqual(self.MIMO_linear.get_input()[1], np.array([0]));
        self.assertEqual(self.MIMO_nonlinear.get_input()[0], np.array([0]));
        self.assertEqual(self.MIMO_nonlinear.get_input()[1], np.array([0]));
        self.assertEqual(self.MIMO_time_varying.get_input()[0], np.array([0]));
        self.assertEqual(self.MIMO_time_varying.get_input()[1], np.array([0]));
    
        self.assertEqual(self.MIMO_linear.get_state()[0], np.array([0]));
        self.assertEqual(self.MIMO_linear.get_state()[1], np.array([0]));
        self.assertEqual(self.MIMO_linear.get_state()[2], np.array([0]));
        self.assertEqual(self.MIMO_nonlinear.get_state()[0], np.array([0]));
        self.assertEqual(self.MIMO_nonlinear.get_state()[1], np.array([0]));
        self.assertEqual(self.MIMO_nonlinear.get_state()[2], np.array([0]));
        self.assertEqual(self.MIMO_time_varying.get_state()[0], np.array([0]));
        self.assertEqual(self.MIMO_time_varying.get_state()[1], np.array([0]));
        self.assertEqual(self.MIMO_time_varying.get_state()[2], np.array([0]));

        self.assertEqual(self.MIMO_linear.get_output()[0], np.array([0]));
        self.assertEqual(self.MIMO_linear.get_output()[1], np.array([0]));
        self.assertEqual(self.MIMO_nonlinear.get_output()[0], np.array([1]));
        self.assertEqual(self.MIMO_nonlinear.get_output()[1], np.array([0]));
        self.assertEqual(self.MIMO_time_varying.get_output()[0], np.array([1]));
        self.assertEqual(self.MIMO_time_varying.get_output()[1], np.array([0]));

    def test_setters(self) -> None:

        """
        
        """

        # SISO models

        self.linear_model.set_input(1);
        self.nonlinear_model.set_input(1);
        self.time_varying_model.set_input(1);

        self.assertEqual(self.linear_model.get_input(), np.array([1]));
        self.assertEqual(self.nonlinear_model.get_input(), np.array([1]));
        self.assertEqual(self.time_varying_model.get_input(), np.array([1]));
    
        # MIMO models

        self.MIMO_linear.set_input(np.array([[1], [3]]));
        self.MIMO_nonlinear.set_input(np.array([[1], [3]]));
        self.MIMO_time_varying.set_input(np.array([[1], [3]]));

        self.assertEqual(self.MIMO_linear.get_input()[0], np.array([1]));
        self.assertEqual(self.MIMO_linear.get_input()[1], np.array([3]));
        self.assertEqual(self.MIMO_nonlinear.get_input()[0], np.array([1]));
        self.assertEqual(self.MIMO_nonlinear.get_input()[1], np.array([3]));
        self.assertEqual(self.MIMO_time_varying.get_input()[0], np.array([1]));
        self.assertEqual(self.MIMO_time_varying.get_input()[1], np.array([3]));

    def test_update(self) -> None:

        """
        
        """

        pass

    def test_eval(self) -> None:

        """
        
        """

        # SISO models

        self.assertEqual(self.linear_model.eval(0.0, self.linear_model.get_state())[0], np.array([0]));
        self.assertEqual(self.linear_model.eval(0.0, self.linear_model.get_state())[1], np.array([0]));
        self.assertEqual(self.linear_model.eval(0.0, self.linear_model.get_state())[2], np.array([0]));
        self.assertEqual(self.linear_model.eval(1.0, self.linear_model.get_state())[0], np.array([0]));
        self.assertEqual(self.linear_model.eval(1.0, self.linear_model.get_state())[1], np.array([0]));
        self.assertEqual(self.linear_model.eval(1.0, self.linear_model.get_state())[2], np.array([0]));

        self.assertEqual(self.nonlinear_model.eval(0.0, self.nonlinear_model.get_state())[0], np.array([2]));
        self.assertEqual(self.nonlinear_model.eval(0.0, self.nonlinear_model.get_state())[1], np.array([0]));
        self.assertEqual(self.nonlinear_model.eval(1.0, self.nonlinear_model.get_state())[0], np.array([2]));
        self.assertEqual(self.nonlinear_model.eval(1.0, self.nonlinear_model.get_state())[1], np.array([0]));

        self.assertEqual(self.time_varying_model.eval(0.0, self.time_varying_model.get_state())[0], np.array([2]));
        self.assertEqual(self.time_varying_model.eval(0.0, self.time_varying_model.get_state())[1], np.array([0]));
        self.assertEqual(self.time_varying_model.eval(1.0, self.time_varying_model.get_state())[0], np.array([2]));
        self.assertEqual(self.time_varying_model.eval(1.0, self.time_varying_model.get_state())[1], np.array([3]));

        #self.assertEqual(self.time_varying_model.eval(0.0, self.time_varying_model.get_state()), np.array([0, 0, 0]).T);

        # MIMO models

        self.assertEqual(self.MIMO_linear.eval(0.0, self.MIMO_linear.get_state())[0], np.array([0]));
        self.assertEqual(self.MIMO_linear.eval(0.0, self.MIMO_linear.get_state())[1], np.array([0]));
        self.assertEqual(self.MIMO_linear.eval(0.0, self.MIMO_linear.get_state())[2], np.array([0]));
        self.assertEqual(self.MIMO_linear.eval(1.0, self.MIMO_linear.get_state())[0], np.array([0]));
        self.assertEqual(self.MIMO_linear.eval(1.0, self.MIMO_linear.get_state())[1], np.array([0]));
        self.assertEqual(self.MIMO_linear.eval(1.0, self.MIMO_linear.get_state())[2], np.array([0]));

        self.assertEqual(self.MIMO_nonlinear.eval(0.0, self.MIMO_nonlinear.get_state())[0], np.array([2]));
        self.assertEqual(self.MIMO_nonlinear.eval(0.0, self.MIMO_nonlinear.get_state())[1], np.array([0]));
        self.assertEqual(self.MIMO_nonlinear.eval(0.0, self.MIMO_nonlinear.get_state())[2], np.array([1]));
        self.assertEqual(self.MIMO_nonlinear.eval(1.0, self.MIMO_nonlinear.get_state())[0], np.array([2]));
        self.assertEqual(self.MIMO_nonlinear.eval(1.0, self.MIMO_nonlinear.get_state())[1], np.array([0]));
        self.assertEqual(self.MIMO_nonlinear.eval(1.0, self.MIMO_nonlinear.get_state())[2], np.array([1]));

        self.assertEqual(self.MIMO_time_varying.eval(0.0, self.MIMO_time_varying.get_state())[0], np.array([2]));
        self.assertEqual(self.MIMO_time_varying.eval(0.0, self.MIMO_time_varying.get_state())[1], np.array([0]));
        self.assertEqual(self.MIMO_time_varying.eval(0.0, self.MIMO_time_varying.get_state())[2], np.array([1]));
        self.assertEqual(self.MIMO_time_varying.eval(1.0, self.MIMO_time_varying.get_state())[0], np.array([2]));
        self.assertEqual(self.MIMO_time_varying.eval(1.0, self.MIMO_time_varying.get_state())[1], np.array([3]));
        self.assertEqual(self.MIMO_time_varying.eval(1.0, self.MIMO_time_varying.get_state())[2], np.array([1]));

if __name__ == "__main__":

    unittest.main();