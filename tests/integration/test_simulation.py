import unittest
from pynamics.simulations import Sim
from pynamics.models.state_space_models import LinearModel
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
        
        stem = "C:/Users/ml4/Desktop/Projects/Repos/pynamics";
    
        files_10sec = ["ode1_test_10sec.csv", "ode2_test_10sec.csv", "ode4_test_10sec.csv"];
        files_100sec = ["ode1_test_100sec.csv", "ode2_test_100sec.csv", "ode4_test_100sec.csv"];
    
        data_10sec = [];
        data_100sec = [];
    
        for (sim_10sec, sim_100sec) in zip(files_10sec, files_100sec):

            path_10sec = f"{stem}/tests/integration/solver_validation_data/{sim_10sec}";
            path_100sec = f"{stem}/tests/integration/solver_validation_data/{sim_100sec}";

            data_10sec.append(pd.read_csv(path_10sec));
            data_100sec.append(pd.read_csv(path_100sec));

        cls.data_10sec = data_10sec;
        cls.data_100sec = data_100sec;
    
        A = np.array([[0, 0, -1], [1, 0, -3], [0, 1, -3]]);
        B = np.array([1, -5, 1]).reshape(-1, 1);
        C = np.array([0, 0, 1]);
        D = np.array([0]);
    
        cls.model1 = LinearModel(np.zeros((3, 1)), np.array([1]), A, B, C, D);
        cls.model2 = LinearModel(np.zeros((3, 1)), np.array([1]), A, B, C, D);
        cls.model3 = LinearModel(np.zeros((3, 1)), np.array([1]), A, B, C, D);
        cls.model4 = LinearModel(np.zeros((3, 1)), np.array([1]), A, B, C, D);
        cls.model5 = LinearModel(np.zeros((3, 1)), np.array([1]), A, B, C, D);
        cls.model6 = LinearModel(np.zeros((3, 1)), np.array([1]), A, B, C, D);
        reference = np.ones(int(10/0.001)+1);
        reference_100sec = np.ones(int(100/0.001)+1);
    
        cls.Euler_10sec = Sim(cls.model1, reference, solver="Euler");
        cls.Heun_10sec = Sim(cls.model2, reference, solver="Heun");
        cls.RK4_10sec = Sim(cls.model3, reference);
    
        cls.Euler_100sec = Sim(cls.model4, reference_100sec, solver="Euler", tfinal=100);
        cls.Heun_100sec = Sim(cls.model5, reference_100sec, solver="Heun", tfinal=100);
        cls.RK4_100sec = Sim(cls.model6, reference_100sec, tfinal=100);

    @classmethod
    def tearDownClass(cls) -> None:
        
        """
        _summary_

        _extended_summary_
        """

        del cls.data_10sec;
        del cls.data_100sec;
        del cls.model1;
        del cls.model2;
        del cls.model3;
        del cls.model4;
        del cls.model5;
        del cls.model6;
        del cls.Euler_10sec;
        del cls.Heun_10sec;
        del cls.RK4_10sec;
        del cls.Euler_100sec;
        del cls.Heun_100sec;
        del cls.RK4_100sec;


    def test_simulations(self) -> None:

        """
        _summary_

        _extended_summary_
        """

        Euler_10sec_res = self.Euler_10sec.run();
        Euler_10sec_true = self.data_10sec[0];
        Sim.tracking_plot(Euler_10sec_res, "Time", "Ref_1", "y_1");
        Sim.system_outputs_plot(Euler_10sec_res, "Time", ["y_1"]);
        Heun_10sec_res = self.Heun_10sec.run();
        #print(Heun_10sec_res);
        RK4_10sec_res = self.RK4_10sec.run();
        #print(RK4_10sec_res);

        Euler_100sec_res = self.Euler_100sec.run();
        #print(Euler_100sec_res);
        Heun_100sec_res = self.Heun_100sec.run();
        RK4_100sec_res = self.RK4_100sec.run();
        RK4_100sec_true = self.data_100sec[2];
        plt.plot(RK4_100sec_true["t"], RK4_100sec_true["y"], label="Matlab");
        plt.plot(RK4_100sec_res["Time"], RK4_100sec_res["y_1"], label="pynamics");
        plt.legend();
        plt.show();
        #print(RK4_100sec_res);

if __name__ == "__main__":

    unittest.main();