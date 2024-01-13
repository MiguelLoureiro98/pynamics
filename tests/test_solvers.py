import unittest
import numpy as np
import pandas as pd
from pynamics.solvers.fixed_step._fixed_step_solvers import Euler, Modified_Euler, Heun, RK4
from pynamics.solvers.variable_step._variable_step_solvers import RKF
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

    def tearDown(self) -> None:
        
        pass

    def test_initialisation(self) -> None:

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

        print("Testing variable step solvers' 'get_step_size' method ...");

        self.assertEqual(self.RKF_solver.get_step_size(), self.h);

    def test_update_step_size(self) -> None:

        """
        Tests every possible case ... .
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

        #model = linearModel(np.zeros((3, 1)), np.zeros((3, 1)));
        file = "ode45_test_10sec.csv";
        path = f"J:/Projectos_e_relatorios/Project_repos/Pynamics/Pynamics/data/solver_validation_data/{file}";
        example_data = pd.read_csv(path);
        print(example_data.describe());

if __name__ == "__main__":

    unittest.main();