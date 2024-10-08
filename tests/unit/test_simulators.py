import unittest
import numpy as np
from pynamics.models.state_space_models import LinearModel
from pynamics.simulations import Sim
from pynamics.controllers.dummy import DummyController

"""
Unit tests for the Sim class.
"""

class Controller_test(object):

    def __init__(self, n_inputs: int, n_outputs: int, sampling_time: int | float) -> None:

        self.Ts = sampling_time;
        self.input_dim = n_inputs;
        self.output_dim = n_outputs;

        return;

    def control(self, ref: np.ndarray, y: np.ndarray) -> np.ndarray:

        return np.array([[1]]);

class TestSimulators(unittest.TestCase):
    """
    Test simulator class.
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
        Set up models, controller, noise and optional values.
        """

        A = np.array([[0, 0, -1], [1, 0, -3], [0, 1, -3]]);
        B = np.array([1, -5, 1]).reshape(-1, 1);
        C = np.array([0, 0, 1]);
        D = np.array([0]);

        self.controller = Controller_test(1, 1, 1.0);
        self.noise_power = 100;

        self.model = LinearModel(np.zeros((3, 1)), np.array([0]), A, B, C, D);
        self.simulation = Sim(self.model, np.zeros(int(10.0/0.001)+1));
        self.controlled_simulation = Sim(self.model, np.zeros(int(10.0/0.001)+1), mode="closed_loop", controller=self.controller);
        self.noisy_simulation = Sim(self.model, np.zeros(int(10.0/0.001)+1), noise_power=self.noise_power);
        self.sim_to_reset = Sim(self.model, np.zeros(int(10.0/0.001)+1));
        self.sim_to_reset.outputs = np.ones(int(10.0/0.001)+1);
        self.sim_to_reset.control_actions = np.ones(int(10.0/0.001)+1);
        self.sim_props = Sim(self.model, np.zeros(int(10.0/0.001)+1));
    
        self.t0 = 0.0;
        self.tfinal = 10.0;
        self.solver = "RK4";
        self.step_size = 0.001;

        return;

    def tearDown(self) -> None:
        """
        Delete variables created by the setUp method.
        """

        del self.noise_power;
        del self.model;
        del self.simulation;
        del self.controlled_simulation;
        del self.noisy_simulation;
        del self.t0;
        del self.tfinal;
        del self.solver;
        del self.step_size;
        del self.sim_to_reset;
        del self.sim_props;

        print("Sucessfully deleted every instance of every class created by the setUp method.");
    
        return;

    def test_initialisation(self) -> None:
        """
        Test initialisation (exceptions and attributes).
        """

        ref_signal_length = int(10.0/0.001) + 1;

        reference = np.zeros(ref_signal_length);
        labelled_sim = Sim(self.model, reference, reference_labels=["Reference_signal"]);

        # Exceptions

        self.assertRaises(TypeError, Sim, 2, reference);
        self.assertRaises(TypeError, Sim, self.model, reference, t0="Hey");
        self.assertRaises(TypeError, Sim, self.model, reference, tfinal="Hey");
        self.assertRaises(TypeError, Sim, self.model, reference, step_size="Hey");
        self.assertRaises(TypeError, Sim, self.model, reference, solver=2.5);
        self.assertRaises(ValueError, Sim, self.model, reference, solver="Not_Implemented");
    
        self.assertRaises(TypeError, Sim, self.model, reference, mode=1);
        self.assertRaises(ValueError, Sim, self.model, reference, mode="Third_way");
        self.assertRaises(TypeError, Sim, self.model, 23.19);
        self.assertRaises(ValueError,Sim, self.model, np.zeros((2, ref_signal_length)));
        self.assertRaises(ValueError, Sim, self.model, np.zeros(5));
        self.assertRaises(TypeError, Sim, self.model, reference, reference_labels=2);
        self.assertRaises(ValueError, Sim, self.model, reference, reference_labels=["Label_1", "Label_2"]);
        self.assertRaises(TypeError, Sim, self.model, reference, reference_lookahead=5.5);
        self.assertRaises(ValueError, Sim, self.model, reference, reference_lookahead=0);
    
        # Attributes
    
        self.assertEqual(self.simulation.options["t0"], self.t0);
        self.assertEqual(self.simulation.options["tfinal"], self.tfinal);
        self.assertEqual(self.simulation.solver.t, self.t0);
        self.assertEqual(self.simulation.solver.h, self.step_size);
        self.assertEqual(self.simulation.time.shape[0], ref_signal_length);

        self.assertEqual(self.simulation.inputs.shape[0], 1);
        self.assertEqual(self.simulation.inputs.shape[1], ref_signal_length);
        self.assertEqual(self.simulation.outputs.shape[0], 1);
        self.assertEqual(self.simulation.outputs.shape[1], ref_signal_length);
        self.assertEqual(self.simulation.control_actions.shape[0], 1);
        self.assertEqual(self.simulation.control_actions.shape[1], ref_signal_length);
        self.assertEqual(isinstance(self.simulation.controller, DummyController), True);
        self.assertEqual(isinstance(self.controlled_simulation.controller, DummyController), False);
        self.assertListEqual(self.simulation.ref_labels, ["Ref_1"]);
        self.assertEqual(self.simulation.noise[0, 0], self.simulation.inputs[0, 0]);
        self.assertEqual(self.simulation.ref_lookahead, 1);

        self.assertListEqual(labelled_sim.ref_labels, ["Reference_signal"]);
        self.assertEqual(self.noisy_simulation.noise.shape[1], self.noisy_simulation.time.shape[0]);
        self.assertNotEqual(self.noisy_simulation.noise[0, 0], self.noisy_simulation.inputs[0, 0]);

    def test_step(self) -> None:
        """
        Test the _step method.
        """

        matlab_output = 0.000996003664875;
        
        sim_outputs, sim_control_actions = self.simulation._step(self.simulation.time[0], self.simulation.inputs[:, 0], self.simulation.outputs[:, 0] + self.simulation.noise[:, 0]);
        controlled_outputs, controlled_control_actions = self.controlled_simulation._step(self.simulation.time[0], self.simulation.inputs[:, 0], self.simulation.outputs[:, 0] + self.simulation.noise[:, 0]);
        noisy_outputs, noisy_control_actions = self.noisy_simulation._step(self.simulation.time[0], self.simulation.inputs[:, 0], self.simulation.outputs[:, 0] + self.simulation.noise[:, 0]);

        self.assertListEqual(sim_outputs.tolist(), np.array([[0]]).tolist());
        self.assertListEqual(controlled_outputs.tolist(), np.array([[matlab_output]]).tolist());
        self.assertNotEqual(sim_outputs[0].item(), noisy_outputs[0].item());

        self.assertEqual(sim_control_actions[0], 0.0);
        self.assertEqual(controlled_control_actions[0], 1.0);
        self.assertEqual(noisy_control_actions[0], 0.0);

    def test_properties(self) -> None:
        """
        Test Sim properties.
        """

        np.testing.assert_array_equal(self.sim_props.inputs, np.zeros((1, int(10.0/0.001)+1)));
        self.sim_props.inputs = np.ones(int(10.0/0.001)+1);
        np.testing.assert_array_equal(self.sim_props.inputs, np.ones((1, int(10.0/0.001)+1)));

        self.assertEqual(self.sim_props.ref_lookahead, 1);
        self.sim_props.ref_lookahead = 5;
        self.assertEqual(self.sim_props.ref_lookahead, 5);

        return;

    def test_reset(self) -> None:
        """
        Test reset() method.
        """

        self.sim_to_reset.reset(np.ones((3, 1)), np.array([1]));

        np.testing.assert_array_equal(self.sim_to_reset.system.x, np.ones((3, 1)));
        np.testing.assert_array_equal(self.sim_to_reset.system.get_input(), np.array([[1]]));
        np.testing.assert_array_equal(self.sim_to_reset.outputs, np.zeros((1, int(10.0/0.001)+1)));
        np.testing.assert_array_equal(self.sim_to_reset.control_actions, np.zeros((1, int(10.0/0.001)+1)));

        return;

if __name__ == "__main__":

    unittest.main();