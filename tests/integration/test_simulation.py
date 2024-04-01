import pandas as pd
import matplotlib.pyplot as plt

file = "ode45_test_10sec.csv";
home = True;

if home is True:

    stem = "J:/Projectos_e_relatorios/Project_repos/Pynamics/";
        
else:

    stem = "C:/Users/User/Desktop/Project_repos/Pynamics/";

path = f"{stem}Pynamics/tests/integration/solver_validation_data/{file}";
example_data = pd.read_csv(path);
plt.plot(example_data["t"], example_data["y"]);
plt.show();
#print(example_data.describe());