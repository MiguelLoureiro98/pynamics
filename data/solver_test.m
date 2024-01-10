% Script to run simulations for an LTI system using the solvers implemented
%   in the "Pynamics" package.
% The results from these simulations will be used to validate the solvers'
%   implementation.

clear
clc

% Define model parameters.
A = [0, 0, -1; 1, 0, -3; 0, 1, -3];
B = [1; -5; 1];
C = [0, 0, 1];
D = 0;

% Define simulation parameters
stop_time = "100.0";
solver = ["ode1", "ode2", "ode4", "ode45"];
fixed_step_solvers = 3;
fixed_step_size = "0.001";
fixed = [true, true, true, false];
results = zeros(str2double(stop_time) / str2double(fixed_step_size) + 1, ...
                2 * fixed_step_solvers);

% Set parameters and run simulations.
test_model = "test_solver";
load_system(test_model);
set_param(test_model, "StopTime", stop_time);

for i=1:length(solver)

    set_param(test_model, "Solver", solver(i));
    
    if(fixed(i) == true)
    
        set_param(test_model, "FixedStep", fixed_step_size);
        sim_results = sim(test_model);
    
        results(:, i) = sim_results.tout;
        results(:, i+1) = sim_results.y;

    end
    
    res_table = table(sim_results.tout, sim_results.y, ...
                      "VariableNames", ["t", "y"]);

    writetable(res_table, solver(i) + "_test.csv");

    plot(sim_results.tout, sim_results.y), hold on;

end

grid on, title("Simulation results"), xlabel("t (s)"), ylabel("y"), ...
    legend("ode1", "ode2", "ode4", "ode45");