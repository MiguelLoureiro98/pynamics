import unittest
import numpy as np
from pynamics.models.state_space_models import linearModel
from pynamics.simulations import sim
from pynamics._controllers._dummy import dummy_controller

"""

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

        A = np.array([[0, 0, -1], [1, 0, -3], [0, 1, -3]]);
        B = np.array([1, -5, 1]).reshape(-1, 1);
        C = np.array([0, 0, 1]);
        D = np.array([0]);

        self.controller = Controller_test(1, 1, 1.0);

        self.model = linearModel(np.zeros((3, 1)), np.array([0]), A, B, C, D);
        self.simulation = sim(self.model, np.zeros(int(10.0/0.001)));
        self.controlled_simulation = sim(self.model, np.zeros(int(10.0/0.001)), mode="closed_loop", controller=self.controller);
    
        self.t0 = 0.0;
        self.tfinal = 10.0;
        self.solver = "RK4";
        self.step_size = 0.001;

    def tearDown(self) -> None:
        
        """
        
        """

        del self.model;
        del self.simulation;
        del self.controlled_simulation;
        del self.t0;
        del self.tfinal;
        del self.solver;
        del self.step_size;

        print("Sucessfully deleted every instance of every class created by the setUp method.");

    def test_initialisation(self) -> None:

        """
        
        """

        reference = np.zeros(int(10.0/0.001));

        # Exceptions

        self.assertRaises(TypeError, sim, 2, reference);
        self.assertRaises(TypeError, sim, self.model, reference, t0="Hey");
        self.assertRaises(TypeError, sim, self.model, reference, tfinal="Hey");
        self.assertRaises(TypeError, sim, self.model, reference, step_size="Hey");
        self.assertRaises(TypeError, sim, self.model, reference, solver=2.5);
        self.assertRaises(ValueError, sim, self.model, reference, solver="Not_Implemented");
    
        self.assertRaises(TypeError, sim, self.model, reference, mode=1);
        self.assertRaises(ValueError, sim, self.model, reference, mode="Third_way");
        self.assertRaises(TypeError, sim, self.model, 23.19);
        self.assertRaises(ValueError, sim, self.model, np.zeros((2, int(10.0/0.001))));
        self.assertRaises(ValueError, sim, self.model, np.zeros(5));
        self.assertRaises(TypeError, sim, self.model, reference, reference_labels=2);
        self.assertRaises(ValueError, sim, self.model, reference, reference_labels=["Label_1", "Label_2"]);
    
        # Attributes
    
        self.assertEqual(self.simulation.options["t0"], self.t0);
        self.assertEqual(self.simulation.options["tfinal"], self.tfinal);
        self.assertEqual(self.simulation.solver.t, self.t0);
        self.assertEqual(self.simulation.solver.h, self.step_size);
        self.assertEqual(self.simulation.time.shape[0], 10000);

        self.assertEqual(self.simulation.inputs.shape[0], 1);
        self.assertEqual(self.simulation.inputs.shape[1], 10000);
        self.assertEqual(self.simulation.outputs.shape[0], 1);
        self.assertEqual(self.simulation.outputs.shape[1], 10000);
        self.assertEqual(self.simulation.control_actions.shape[0], 1);
        self.assertEqual(self.simulation.control_actions.shape[1], 10000);
        self.assertEqual(isinstance(self.simulation.controller, dummy_controller), True);
        self.assertListEqual(self.simulation.ref_labels, ["Ref_1"]);

    def test_step(self) -> None:

        """
        
        """

        pass

if __name__ == "__main__":

    unittest.main();