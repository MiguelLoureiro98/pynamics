import unittest
import numpy as np
import pandas as pd
from pynamics.solvers.fixed_step._fixed_step_solvers import Euler, Modified_Euler, Heun, RK4
from pynamics.solvers.variable_step._variable_step_solvers import RKF, DP
from pynamics.models.state_space_models import linearModel

"""
Test cases to test every single solver supported by this package.
"""

class TestSolvers(unittest.TestCase):

    """
    Generic test class to implement the test cases for all solvers.
    """

    @classmethod
    def setUpClass(cls) -> None:
        
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        
        pass

    def setUp(self) -> None:

        """
        Define testing parameters.
        """

        self.sig_digits = 5;
        self.h = 0.1;
        self.initial_t = 0.0;
        self.final_t = 100.0;
        self.time_testing = np.round(self.final_t - 0.05, self.sig_digits);
        self.max_min_residue = 0.00000001;
        self.min_h = 0.01;
        self.max_h = 10;
        self.min_q = 0.1;
        self.max_q = 4;
        self.time_seq = np.round(np.arange(self.initial_t + self.h, self.final_t, self.h), self.sig_digits);
        self.euler_solver = Euler(self.h, self.initial_t);
        self.RK2_solver = Modified_Euler(self.h, self.initial_t);
        self.RK3_solver = Heun(self.h, self.initial_t);
        self.RK4_solver = RK4(self.h, self.initial_t);
        self.RKF_solver = RKF(self.h, self.initial_t, max_step_size=self.max_h, min_step_size=self.min_h, 
                              min_update=self.min_q, max_update=self.max_q, tfinal=self.final_t);
        self.DP_solver = DP(self.h, self.initial_t, max_step_size=self.max_h, min_step_size=self.min_h, 
                              min_update=self.min_q, max_update=self.max_q, tfinal=self.final_t);
    

        # This should be moved to the setUpClass method.
        self.sim_h = 0.001;
        self.sim_hmin = 1e-5;
        self.ode1 = Euler(self.sim_h, self.initial_t);
        self.ode2 = Heun(self.sim_h, self.initial_t);
        self.ode4 = RK4(self.sim_h, self.initial_t);
        self.ode_RKF = RKF(self.sim_h, self.initial_t, max_step_size=self.max_h, min_step_size=self.sim_hmin, 
                           min_update=1e-15, max_update=10000, tfinal=self.final_t);
        self.ode_DP = DP(self.sim_h, self.initial_t, max_step_size=self.max_h, min_step_size=self.sim_hmin, 
                            min_update=1e-15, max_update=10000, tfinal=self.final_t);

    def tearDown(self) -> None:
        
        pass

    def test_initialisation(self) -> None:

        """
        Tests the initialisation procedure for every supported solver.
        """

        print("Testing class initialisation methods ...");

        self.assertEqual(self.euler_solver.get_time_step(), self.initial_t);
        self.assertEqual(self.RK2_solver.get_time_step(), self.initial_t);
        self.assertEqual(self.RK3_solver.get_time_step(), self.initial_t);
        self.assertEqual(self.RK4_solver.get_time_step(), self.initial_t);
        self.assertEqual(self.RKF_solver.get_time_step(), self.initial_t);
    
        self.assertEqual(self.euler_solver.h, self.h);
        self.assertEqual(self.RK2_solver.h, self.h);
        self.assertEqual(self.RK3_solver.h, self.h);
        self.assertEqual(self.RK4_solver.h, self.h);
        self.assertEqual(self.RKF_solver.get_step_size(), self.h);
        
    def test_update_time_step(self) -> None:

        """
        
        """

        print("Testing 'update_time_step' method ...");

        for n in self.time_seq:

            self.euler_solver._update_time_step();
            self.RK2_solver._update_time_step();
            self.RK3_solver._update_time_step();
            self.RK4_solver._update_time_step();
            self.RKF_solver._update_time_step();

            self.assertEqual(np.round(self.euler_solver.get_time_step(), self.sig_digits), n);
            self.assertEqual(np.round(self.RK2_solver.get_time_step(), self.sig_digits), n);
            self.assertEqual(np.round(self.RK3_solver.get_time_step(), self.sig_digits), n);
            self.assertEqual(np.round(self.RK4_solver.get_time_step(), self.sig_digits), n);
            self.assertEqual(np.round(self.RKF_solver.get_time_step(), self.sig_digits), n);

    def test_get_step_size(self) -> None:

        """
        
        """

        print("Testing variable step solvers' 'get_step_size' method ...");

        self.assertEqual(self.RKF_solver.get_step_size(), self.h);

    def test_update_step_size(self) -> None:

        """
        Tests the update_step_size method of variable-step solvers.

        6 cases are tested:
        1 - The step size should stay the same;
        2 - The step size update should be clipped to its minimum value;
        3 - The step size update should be clipped to its maximum value;
        4 - The step size should be clipped to the minimum step size;
        5 - The step size should be clipped to the maximum step size;
        6 - The step size should be the difference between the final time step and the current time step.
        """

        print("Testing variable step solvers' 'update_step_size' method ...");

        q = [1, self.min_q - 1, self.max_q + 1, self.min_q, self.max_q, 1];
        val = [self.h, np.round(self.min_q * self.h, self.sig_digits), np.round(self.max_q * self.h, self.sig_digits), self.min_h, self.max_h, 
               np.round(self.final_t - self.time_testing, self.sig_digits)];

        self.RKF_solver._update_step_size(q[0]);
        self.assertEqual(np.round(self.RKF_solver.h, self.sig_digits), val[0]);
        self.RKF_solver.h = self.h;
    
        self.RKF_solver._update_step_size(q[1]);
        self.assertEqual(np.round(self.RKF_solver.h, self.sig_digits), val[1]);
        self.RKF_solver.h = self.h;
    
        self.RKF_solver._update_step_size(q[2]);
        self.assertEqual(np.round(self.RKF_solver.h, self.sig_digits), val[2]);
        self.RKF_solver.h = self.min_h + self.max_min_residue;
    
        self.RKF_solver._update_step_size(q[3]);
        self.assertEqual(np.round(self.RKF_solver.h, self.sig_digits), val[3]);
        self.RKF_solver.h = self.max_h - self.max_min_residue;
    
        self.RKF_solver._update_step_size(q[4]);
        self.assertEqual(np.round(self.RKF_solver.h, self.sig_digits), val[4]);
        self.RKF_solver.h = self.h;
    
        self.RKF_solver.t = self.time_testing;
        self.RKF_solver._update_step_size(q[5]);
        self.assertEqual(np.round(self.RKF_solver.tfinal - self.RKF_solver.t, self.sig_digits), val[5]);
        self.assertEqual(np.round(self.RKF_solver.h, self.sig_digits), val[5]);
        self.RKF_solver.h = self.h;

    def test_step(self) -> None:

        """
        
        """

        print("Testing step method ...");

        A = np.array([[0, 0, -1], [1, 0, -3], [0, 1, -3]]);
        B = np.array([1, -5, 1]).reshape(-1, 1);
        C = np.array([0, 0, 1]);
        D = np.array([0]);
        model = linearModel(np.zeros((3, 1)), np.array([1]), A, B, C, D);

        ode1_matlab = 0.001;
        ode2_matlab = 0.000996;
        ode4_matlab = np.round(0.000996003664875, self.sig_digits + 3);
        ode45_matlab = np.round(0.000996003664875, self.sig_digits + 3);

        ode1_output = np.matmul(C, self.ode1.step(model)).item();
        ode2_output = np.matmul(C, self.ode2.step(model)).item();
        ode4_output = np.matmul(C, self.ode4.step(model)).item();
        next_state = self.ode_DP.step(model);
        odeRKF_output = np.matmul(C, next_state).item();

        #print(odeRKF_output);
        #print(self.ode_RKF.get_step_size());

        for it in np.arange(0, 2):

            print(self.ode_DP.h);
            model.update_state(state=next_state);
            next_next_state = self.ode_DP.step(model);
            new_odeRKF_output = np.matmul(C, next_next_state).item();
        
        print(self.ode_DP.t);

        self.assertEqual(np.round(ode1_output, self.sig_digits + 3), ode1_matlab);
        self.assertEqual(np.round(ode2_output, self.sig_digits + 3), ode2_matlab);
        self.assertEqual(np.round(ode4_output, self.sig_digits + 3), ode4_matlab);
        #self.assertEqual(np.round(odeRKF_output, self.sig_digits), ode45_matlab);

if __name__ == "__main__":

    unittest.main();