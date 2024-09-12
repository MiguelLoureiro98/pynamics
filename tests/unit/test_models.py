import unittest
import numpy as np
from pynamics.models.state_space_models import LinearModel, NonlinearModel

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

        """
        
        """

        # SISO models

        A = np.array([[0, 0, -1], [1, 0, -3], [0, 1, -3]]);
        B = np.array([1, -5, 1]).reshape(-1, 1);
        C = np.array([0, 0, 1]);
        D = np.array([0]);
        linmodel = LinearModel(np.zeros((3, 1)), np.array([0]), A, B, C, D);

        nlinmodel = NonlinearModel(np.zeros((2, 1)), np.array([0]), state_function_test, output_function_test, 1, 1);
        time_variant_model = NonlinearModel(np.zeros((2, 1)), np.array([0]), time_dependent_state_function_test, output_function_test, 1, 1);

        self.linear_model = linmodel;
        self.nonlinear_model = nlinmodel;
        self.time_varying_model = time_variant_model;
    
        # MIMO models
    
        Bm = np.array([[1, 1], [-5, 0], [1, 0]]);
        Cm = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
        Dm = np.array([[0, 0], [0, 0], [0, 1]]);
        MIMO_linmodel = LinearModel(np.zeros((3, 1)), np.array([[0], [0]]), A, Bm, Cm, Dm);
    
        MIMO_nlinmodel = NonlinearModel(np.zeros((3, 1)), np.array([[0], [0]]), MIMO_state_function_test, MIMO_output_function_test, 2, 2);
        MIMO_time_variant_model = NonlinearModel(np.zeros((3, 1)), np.array([[0], [0]]), MIMO_time_dependent_state_function_test, MIMO_output_function_test, 2, 2);
    
        self.MIMO_linear = MIMO_linmodel;
        self.MIMO_nonlinear = MIMO_nlinmodel;
        self.MIMO_time_varying = MIMO_time_variant_model;

    def tearDown(self) -> None:
        
        """
        
        """

        del self.linear_model
        del self.nonlinear_model;
        del self.time_varying_model;
        del self.MIMO_linear;
        del self.MIMO_nonlinear;
        del self.MIMO_time_varying;

        print("Deleted every testing variable.");

    def test_initialisation(self) -> None:

        """
        
        """

        new_linear = LinearModel(np.zeros((3, 1)), np.array([0]), self.linear_model.A, self.linear_model.B, self.linear_model.C, self.linear_model.D, \
                                 input_labels=["Input"], output_labels=["Output"]);
        new_nonlinear = NonlinearModel(np.zeros((2, 1)), np.array([0]), state_function_test, output_function_test, 1, 1, input_labels=["Input"], output_labels=["Output"]);

        new_MIMO_linear = LinearModel(np.zeros((3, 1)), np.array([[0], [0]]), self.MIMO_linear.A, self.MIMO_linear.B, self.MIMO_linear.C, self.MIMO_linear.D, \
                                      input_labels=["Input_1", "Input_2"], output_labels=["Output_1", "Output_2", "Output_3"]);
        new_MIMO_nonlinear = NonlinearModel(np.zeros((3, 1)), np.array([[0], [0]]), MIMO_state_function_test, MIMO_output_function_test, 2, 2, \
                                            input_labels=["Input_1", "Input_2"], output_labels=["Output_1", "Output_2"]);

        # Parameters

        # SISO models

        self.assertEqual(self.linear_model.input_dim, 1);
        self.assertEqual(self.linear_model.output_dim, 1);
        self.assertEqual(self.linear_model.state_dim, 3);
        self.assertEqual(self.linear_model.input_labels[0], "u_1");
        self.assertEqual(self.linear_model.output_labels[0], "y_1");
    
        self.assertEqual(self.nonlinear_model.input_dim, 1);
        self.assertEqual(self.nonlinear_model.output_dim, 1);
        self.assertEqual(self.nonlinear_model.state_dim, 2);
        self.assertEqual(self.nonlinear_model.input_labels[0], "u_1");
        self.assertEqual(self.nonlinear_model.output_labels[0], "y_1");
    
        self.assertEqual(self.time_varying_model.input_dim, 1);
        self.assertEqual(self.time_varying_model.output_dim, 1);
        self.assertEqual(self.time_varying_model.state_dim, 2);
        self.assertEqual(self.time_varying_model.input_labels[0], "u_1");
        self.assertEqual(self.time_varying_model.output_labels[0], "y_1");

        self.assertListEqual(new_linear.input_labels, ["Input"]);
        self.assertListEqual(new_linear.output_labels, ["Output"]);
        self.assertListEqual(new_nonlinear.input_labels, ["Input"]);
        self.assertListEqual(new_nonlinear.output_labels, ["Output"]);
    
        # MIMO models
    
        self.assertEqual(self.MIMO_linear.input_dim, 2);
        self.assertEqual(self.MIMO_linear.output_dim, 3);
        self.assertEqual(self.MIMO_linear.state_dim, 3);
        self.assertListEqual(self.MIMO_linear.input_labels, ["u_1", "u_2"]);
        self.assertListEqual(self.MIMO_linear.output_labels, ["y_1", "y_2", "y_3"]);

        self.assertEqual(self.MIMO_nonlinear.input_dim, 2);
        self.assertEqual(self.MIMO_nonlinear.output_dim, 2);
        self.assertEqual(self.MIMO_nonlinear.state_dim, 3);
        self.assertListEqual(self.MIMO_nonlinear.input_labels, ["u_1", "u_2"]);
        self.assertListEqual(self.MIMO_nonlinear.output_labels, ["y_1", "y_2"]);

        self.assertEqual(self.MIMO_time_varying.input_dim, 2);
        self.assertEqual(self.MIMO_time_varying.output_dim, 2);
        self.assertEqual(self.MIMO_time_varying.state_dim, 3);
        self.assertListEqual(self.MIMO_time_varying.input_labels, ["u_1", "u_2"]);
        self.assertListEqual(self.MIMO_time_varying.output_labels, ["y_1", "y_2"]);

        self.assertListEqual(new_MIMO_linear.input_labels, ["Input_1", "Input_2"]);
        self.assertListEqual(new_MIMO_linear.output_labels, ["Output_1", "Output_2", "Output_3"]);
        self.assertListEqual(new_MIMO_nonlinear.input_labels, ["Input_1", "Input_2"]);
        self.assertListEqual(new_MIMO_nonlinear.output_labels, ["Output_1", "Output_2"]);

        # Exceptions
    
        self.assertRaises(TypeError, NonlinearModel, np.zeros((2, 1)), np.array([0]), state_function_test, output_function_test, 1.4, 1);
        self.assertRaises(TypeError, NonlinearModel, np.zeros((2, 1)), np.array([0]), state_function_test, output_function_test, 1, 1.7);

        self.assertRaises(TypeError, NonlinearModel, np.zeros((2, 1)), np.array([0]), state_function_test, output_function_test, 1, 1, input_labels=1);
        self.assertRaises(TypeError, NonlinearModel, np.zeros((2, 1)), np.array([0]), state_function_test, output_function_test, 1, 1, output_labels=1);
    
        self.assertRaises(ValueError, NonlinearModel, np.zeros((2, 1)), np.array([0]), state_function_test, output_function_test, 1, 1, input_labels=["1", "2"]);
        self.assertRaises(ValueError, NonlinearModel, np.zeros((2, 1)), np.array([0]), state_function_test, output_function_test, 1, 1, output_labels=["1", "2"]);
    
        self.assertRaises(TypeError, LinearModel, np.zeros((3, 1)), np.array([0]), 1, self.linear_model.B, self.linear_model.C, self.linear_model.D);
        self.assertRaises(TypeError, LinearModel, np.zeros((3, 1)), np.array([0]), self.linear_model.A, 1, self.linear_model.C, self.linear_model.D);
        self.assertRaises(TypeError, LinearModel, np.zeros((3, 1)), np.array([0]), self.linear_model.A, self.linear_model.B, 1, self.linear_model.D);
        self.assertRaises(TypeError, LinearModel, np.zeros((3, 1)), np.array([0]), self.linear_model.A, self.linear_model.B, self.linear_model.C, 1);
    
        self.assertRaises(ValueError, LinearModel, np.zeros((3, 1)), np.array([0]), np.zeros((3, 2)), self.linear_model.B, self.linear_model.C, self.linear_model.D);
        self.assertRaises(ValueError, LinearModel, np.zeros((3, 1)), np.array([0]), np.zeros((4, 4)), self.linear_model.B, self.linear_model.C, self.linear_model.D);
        self.assertRaises(ValueError, LinearModel, np.zeros((3, 1)), np.array([0]), self.linear_model.A, np.zeros((3, 2)), self.linear_model.C, self.linear_model.D);
        self.assertRaises(ValueError, LinearModel, np.zeros((3, 1)), np.array([0]), self.linear_model.A, np.zeros((2, 1)), self.linear_model.C, self.linear_model.D);
        self.assertRaises(ValueError, LinearModel, np.zeros((3, 1)), np.array([0]), self.linear_model.A, self.linear_model.B, np.zeros((2, 3)), self.linear_model.D);
        self.assertRaises(ValueError, LinearModel, np.zeros((3, 1)), np.array([0]), self.linear_model.A, self.linear_model.B, self.linear_model.C, np.zeros((1, 3)));
        self.assertRaises(ValueError, LinearModel, np.zeros((3, 1)), np.array([0]), self.linear_model.A, self.linear_model.B, np.zeros((1, 5)), self.linear_model.D);
    
        self.assertRaises(TypeError, LinearModel, np.zeros((3, 1)), np.array([0]), self.linear_model.A, self.linear_model.B, self.linear_model.C, self.linear_model.D, input_labels=1);
        self.assertRaises(TypeError, LinearModel, np.zeros((3, 1)), np.array([0]), self.linear_model.A, self.linear_model.B, self.linear_model.C, self.linear_model.D, output_labels=1);
    
        self.assertRaises(ValueError, LinearModel, np.zeros((3, 1)), np.array([0]), self.linear_model.A, self.linear_model.B, self.linear_model.C, self.linear_model.D, input_labels=["1", "2"]);
        self.assertRaises(ValueError, LinearModel, np.zeros((3, 1)), np.array([0]), self.linear_model.A, self.linear_model.B, self.linear_model.C, self.linear_model.D, output_labels=["1", "2"]);

    def test_getters(self) -> None:

        """
        
        """

        zero = np.array([0]);
        one = np.array([1]);

        # SISO models

        self.assertEqual(self.linear_model.get_input(), zero);
        self.assertEqual(self.nonlinear_model.get_input(), zero);
        self.assertEqual(self.time_varying_model.get_input(), zero);
    
        self.assertEqual(self.linear_model.get_state()[0], zero);
        self.assertEqual(self.linear_model.get_state()[1], zero);
        self.assertEqual(self.linear_model.get_state()[2], zero);
        #self.assertListEqual(self.linear_model.get_state().tolist(), [0.0, 0.0, 0.0]); -> Don't do this -- formatting problems (unless we flatten the array)
        self.assertEqual(self.nonlinear_model.get_state()[0], zero);
        self.assertEqual(self.nonlinear_model.get_state()[1], zero);
        self.assertEqual(self.time_varying_model.get_state()[0], zero);
        self.assertEqual(self.time_varying_model.get_state()[1], zero);
    
        self.assertEqual(self.linear_model.get_output(), zero);
        self.assertEqual(self.nonlinear_model.get_output(), one);
        self.assertEqual(self.time_varying_model.get_output(), one);
    
        # MIMO models
    
        self.assertEqual(self.MIMO_linear.get_input()[0], zero);
        self.assertEqual(self.MIMO_linear.get_input()[1], zero);
        self.assertEqual(self.MIMO_nonlinear.get_input()[0], zero);
        self.assertEqual(self.MIMO_nonlinear.get_input()[1], zero);
        self.assertEqual(self.MIMO_time_varying.get_input()[0], zero);
        self.assertEqual(self.MIMO_time_varying.get_input()[1], zero);
    
        self.assertEqual(self.MIMO_linear.get_state()[0], zero);
        self.assertEqual(self.MIMO_linear.get_state()[1], zero);
        self.assertEqual(self.MIMO_linear.get_state()[2], zero);
        self.assertEqual(self.MIMO_nonlinear.get_state()[0], zero);
        self.assertEqual(self.MIMO_nonlinear.get_state()[1], zero);
        self.assertEqual(self.MIMO_nonlinear.get_state()[2], zero);
        self.assertEqual(self.MIMO_time_varying.get_state()[0], zero);
        self.assertEqual(self.MIMO_time_varying.get_state()[1], zero);
        self.assertEqual(self.MIMO_time_varying.get_state()[2], zero);

        self.assertEqual(self.MIMO_linear.get_output()[0], zero);
        self.assertEqual(self.MIMO_linear.get_output()[1], zero);
        self.assertEqual(self.MIMO_nonlinear.get_output()[0], one);
        self.assertEqual(self.MIMO_nonlinear.get_output()[1], zero);
        self.assertEqual(self.MIMO_time_varying.get_output()[0], one);
        self.assertEqual(self.MIMO_time_varying.get_output()[1], zero);

    def test_setters(self) -> None:

        """
        
        """

        one = np.array([1]);
        three = np.array([3]);

        # SISO models

        self.linear_model.set_input(1);
        self.nonlinear_model.set_input(1);
        self.time_varying_model.set_input(1);

        self.assertEqual(self.linear_model.get_input(), one);
        self.assertEqual(self.nonlinear_model.get_input(), one);
        self.assertEqual(self.time_varying_model.get_input(), one);
    
        # MIMO models

        self.MIMO_linear.set_input(np.array([[1], [3]]));
        self.MIMO_nonlinear.set_input(np.array([[1], [3]]));
        self.MIMO_time_varying.set_input(np.array([[1], [3]]));

        self.assertEqual(self.MIMO_linear.get_input()[0], one);
        self.assertEqual(self.MIMO_linear.get_input()[1], three);
        self.assertEqual(self.MIMO_nonlinear.get_input()[0], one);
        self.assertEqual(self.MIMO_nonlinear.get_input()[1], three);
        self.assertEqual(self.MIMO_time_varying.get_input()[0], one);
        self.assertEqual(self.MIMO_time_varying.get_input()[1], three);

    def test_update(self) -> None:

        """
        
        """

        new_state = np.array([3, 2, 1]).T;

        self.linear_model.update_state(new_state);
        self.nonlinear_model.update_state(new_state);
        self.time_varying_model.update_state(new_state);
        self.MIMO_linear.update_state(new_state);
        self.MIMO_nonlinear.update_state(new_state);
        self.MIMO_time_varying.update_state(new_state);

        # SISO models
    
        self.assertEqual(self.linear_model.get_state()[0], new_state[0]);
        self.assertEqual(self.linear_model.get_state()[1], new_state[1]);
        self.assertEqual(self.linear_model.get_state()[2], new_state[2]);
        self.assertEqual(self.nonlinear_model.get_state()[0], new_state[0]);
        self.assertEqual(self.nonlinear_model.get_state()[1], new_state[1]);
        self.assertEqual(self.nonlinear_model.get_state()[2], new_state[2]);
        self.assertEqual(self.time_varying_model.get_state()[0], new_state[0]);
        self.assertEqual(self.time_varying_model.get_state()[1], new_state[1]);
        self.assertEqual(self.time_varying_model.get_state()[2], new_state[2]);

        # MIMO models

        self.assertEqual(self.MIMO_linear.get_state()[0], new_state[0]);
        self.assertEqual(self.MIMO_linear.get_state()[1], new_state[1]);
        self.assertEqual(self.MIMO_linear.get_state()[2], new_state[2]);
        self.assertEqual(self.MIMO_nonlinear.get_state()[0], new_state[0]);
        self.assertEqual(self.MIMO_nonlinear.get_state()[1], new_state[1]);
        self.assertEqual(self.MIMO_nonlinear.get_state()[2], new_state[2]);
        self.assertEqual(self.MIMO_time_varying.get_state()[0], new_state[0]);
        self.assertEqual(self.MIMO_time_varying.get_state()[1], new_state[1]);
        self.assertEqual(self.MIMO_time_varying.get_state()[2], new_state[2]);

    def test_eval(self) -> None:

        """
        
        """

        zero = np.array([0]);
        one = np.array([1]);
        two = np.array([2]);
        three = np.array([3]);

        # SISO models

        self.assertEqual(self.linear_model.eval(0.0, self.linear_model.get_state())[0], zero);
        self.assertEqual(self.linear_model.eval(0.0, self.linear_model.get_state())[1], zero);
        self.assertEqual(self.linear_model.eval(0.0, self.linear_model.get_state())[2], zero);
        self.assertEqual(self.linear_model.eval(1.0, self.linear_model.get_state())[0], zero);
        self.assertEqual(self.linear_model.eval(1.0, self.linear_model.get_state())[1], zero);
        self.assertEqual(self.linear_model.eval(1.0, self.linear_model.get_state())[2], zero);

        self.assertEqual(self.nonlinear_model.eval(0.0, self.nonlinear_model.get_state())[0], two);
        self.assertEqual(self.nonlinear_model.eval(0.0, self.nonlinear_model.get_state())[1], zero);
        self.assertEqual(self.nonlinear_model.eval(1.0, self.nonlinear_model.get_state())[0], two);
        self.assertEqual(self.nonlinear_model.eval(1.0, self.nonlinear_model.get_state())[1], zero);

        self.assertEqual(self.time_varying_model.eval(0.0, self.time_varying_model.get_state())[0], two);
        self.assertEqual(self.time_varying_model.eval(0.0, self.time_varying_model.get_state())[1], zero);
        self.assertEqual(self.time_varying_model.eval(1.0, self.time_varying_model.get_state())[0], two);
        self.assertEqual(self.time_varying_model.eval(1.0, self.time_varying_model.get_state())[1], three);

        # MIMO models

        self.assertEqual(self.MIMO_linear.eval(0.0, self.MIMO_linear.get_state())[0], zero);
        self.assertEqual(self.MIMO_linear.eval(0.0, self.MIMO_linear.get_state())[1], zero);
        self.assertEqual(self.MIMO_linear.eval(0.0, self.MIMO_linear.get_state())[2], zero);
        self.assertEqual(self.MIMO_linear.eval(1.0, self.MIMO_linear.get_state())[0], zero);
        self.assertEqual(self.MIMO_linear.eval(1.0, self.MIMO_linear.get_state())[1], zero);
        self.assertEqual(self.MIMO_linear.eval(1.0, self.MIMO_linear.get_state())[2], zero);

        self.assertEqual(self.MIMO_nonlinear.eval(0.0, self.MIMO_nonlinear.get_state())[0], two);
        self.assertEqual(self.MIMO_nonlinear.eval(0.0, self.MIMO_nonlinear.get_state())[1], zero);
        self.assertEqual(self.MIMO_nonlinear.eval(0.0, self.MIMO_nonlinear.get_state())[2], one);
        self.assertEqual(self.MIMO_nonlinear.eval(1.0, self.MIMO_nonlinear.get_state())[0], two);
        self.assertEqual(self.MIMO_nonlinear.eval(1.0, self.MIMO_nonlinear.get_state())[1], zero);
        self.assertEqual(self.MIMO_nonlinear.eval(1.0, self.MIMO_nonlinear.get_state())[2], one);

        self.assertEqual(self.MIMO_time_varying.eval(0.0, self.MIMO_time_varying.get_state())[0], two);
        self.assertEqual(self.MIMO_time_varying.eval(0.0, self.MIMO_time_varying.get_state())[1], zero);
        self.assertEqual(self.MIMO_time_varying.eval(0.0, self.MIMO_time_varying.get_state())[2], one);
        self.assertEqual(self.MIMO_time_varying.eval(1.0, self.MIMO_time_varying.get_state())[0], two);
        self.assertEqual(self.MIMO_time_varying.eval(1.0, self.MIMO_time_varying.get_state())[1], three);
        self.assertEqual(self.MIMO_time_varying.eval(1.0, self.MIMO_time_varying.get_state())[2], one);

if __name__ == "__main__":

    unittest.main();