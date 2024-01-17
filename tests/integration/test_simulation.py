import pandas as pd

file = "ode45_test_10sec.csv";
home = False;
if home is True:

    stem = "J:/Projectos_e_relatorios/Project_repos/Pynamics/";
        
else:

    stem = "C:/Users/User/Desktop/Project_repos/Pynamics/";

path = f"{stem}Pynamics/data/solver_validation_data/{file}";
example_data = pd.read_csv(path);
print(example_data.describe());