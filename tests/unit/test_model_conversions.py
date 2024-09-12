import unittest
from pynamics.models.state_space_models import LinearModel
from pynamics.models.model_conversion import pynamics_to_control, control_to_pynamics
import control as ct
import numpy as np

"""
This file containts unit tests for the model conversion functions.
"""

class TestConversions(unittest.TestCase):

    def setUp(self) -> None:
        """
        Define the necessary variables for the tests.
        """

        self.x0 = np.zeros((3, 1));
        self.u0 = np.array([0]);

        A = np.array([[0, 0, -1], [1, 0, -3], [0, 1, -3]]);
        B = np.array([1, -5, 1]).reshape(-1, 1);
        C = np.array([0, 0, 1]);
        D = np.array([0]);

        self.pynamics_model = LinearModel(self.x0, self.u0, A, B, C, D);
        self.control_model = ct.ss(A, B, C, D);

        return;
    
    def tearDown(self) -> None:
        """
        Delete the variables used for testing.
        """

        del self.pynamics_model;
        del self.control_model;
        del self.x0;
        del self.u0;
    
        return;
    
    def test_conversions(self) -> None:
        """
        Test conversion functions.
        """

        new_control = pynamics_to_control(self.pynamics_model);
        new_pynamics = control_to_pynamics(self.control_model, self.x0, self.u0);

        np.testing.assert_array_equal(new_control.A, self.control_model.A);
        np.testing.assert_array_equal(new_control.B, self.control_model.B);
        np.testing.assert_array_equal(new_control.C, self.control_model.C);
        np.testing.assert_array_equal(new_control.D, self.control_model.D);

        np.testing.assert_array_equal(new_pynamics.A, self.pynamics_model.A);
        np.testing.assert_array_equal(new_pynamics.B, self.pynamics_model.B);
        np.testing.assert_array_equal(new_pynamics.C, self.pynamics_model.C);
        np.testing.assert_array_equal(new_pynamics.D, self.pynamics_model.D);

        return;