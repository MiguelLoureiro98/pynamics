import unittest
from pynamics.simulations import sim
from pynamics.models.state_space_models import linearModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
This file contains integration tests for the simulation classes.
Essentially, simulations are run using these classes, and the results are then compared to those obtained using Matlab (and the same solvers).
"""

#file = "ode45_test_10sec.csv";
#home = True;

#if home is True:

#    stem = "J:/Projectos_e_relatorios/Project_repos/Pynamics/";
        
#else:

#    stem = "C:/Users/User/Desktop/Project_repos/Pynamics/";

#path = f"{stem}Pynamics/tests/integration/solver_validation_data/{file}";
#example_data = pd.read_csv(path);
#plt.plot(example_data["t"], example_data["y"]);
#plt.show();
#print(example_data.describe());

class SimulationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        
        """
        _summary_

        _extended_summary_
        """
        
        home = True;

        if home is True:

            stem = "J:/Projectos_e_relatorios/Project_repos/Pynamics/";
                
        else:

            stem = "C:/Users/User/Desktop/Project_repos/Pynamics/";
    
        files_10sec = ["ode1_test_10sec.csv", "ode2_test_10sec.csv", "ode4_test_10sec.csv"];
        files_100sec = ["ode1_test_100sec.csv", "ode2_test_100sec.csv", "ode4_test_100sec.csv"];
    
        data_10sec = [];
        data_100sec = [];
    
        for (sim_10sec, sim_100sec) in zip(files_10sec, files_100sec):

            path_10sec = f"{stem}Pynamics/tests/integration/solver_validation_data/{sim_10sec}";
            path_100sec = f"{stem}Pynamics/tests/integration/solver_validation_data/{sim_100sec}";

            data_10sec.append(pd.read_csv(path_10sec));
            data_100sec.append(pd.read_csv(path_100sec));

        cls.data_10sec = data_10sec;
        cls.data_100sec = data_100sec;
    
        A = np.array([[0, 0, -1], [1, 0, -3], [0, 1, -3]]);
        B = np.array([1, -5, 1]).reshape(-1, 1);
        C = np.array([0, 0, 1]);
        D = np.array([0]);
    
        cls.model = linearModel(np.zeros((3, 1)), np.array([0]), A, B, C, D);
        reference = np.ones(int(10/0.001));
        reference_100sec = np.ones(int(100/0.001));
    
        cls.Euler_10sec = sim(cls.model, reference, solver="Euler");
        cls.Heun_10sec = sim(cls.model, reference, solver="Heun");
        cls.RK4_10sec = sim(cls.model, reference);
    
        cls.Euler_100sec = sim(cls.model, reference_100sec, solver="Euler", tfinal=100);
        cls.Heun_100sec = sim(cls.model, reference_100sec, solver="Heun", tfinal=100);
        cls.RK4_100sec = sim(cls.model, reference_100sec, tfinal=100);

    @classmethod
    def tearDownClass(cls) -> None:
        
        """
        _summary_

        _extended_summary_
        """

        del cls.data_10sec;
        del cls.data_100sec;

    def test_simulations(self) -> None:

        """
        _summary_

        _extended_summary_
        """

        # Something's wrong with the run method ...

        Euler_10sec_res = self.Euler_10sec.run();
        #print(Euler_10sec_res);
        Heun_10sec_res = self.Heun_10sec.run();
        #print(Heun_10sec_res);
        RK4_10sec_res = self.RK4_10sec.run();
        #print(RK4_10sec_res);

        Euler_100sec_res = self.Euler_100sec.run();
        #print(Euler_100sec_res);
        Heun_100sec_res = self.Heun_100sec.run();
        RK4_100sec_res = self.RK4_100sec.run();

if __name__ == "__main__":

    unittest.main();