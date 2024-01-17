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
stop_time = "10.0";
solver = ["ode1", "ode2", "ode4", "ode45"];
fixed_step_solvers = 3;
fixed_step_size = "0.001";
fixed = [true, true, true, false];
results = zeros(str2double(stop_time) / str2double(fixed_step_size) + 1, ...
                2 * fixed_step_solvers);

% Set parameters and run simulations for 10 second simulations.
test_model = "test_solver";
load_system(test_model);
set_param(test_model, "StopTime", stop_time);
figure

for i=1:length(solver)

    set_param(test_model, "Solver", solver(i));
    
    if(fixed(i) == true)
    
        set_param(test_model, "SolverType", "Fixed-step");
        set_param(test_model, "FixedStep", fixed_step_size);
        sim_results = sim(test_model);
    
        results(:, i) = sim_results.tout;
        results(:, i+1) = sim_results.y;

    else

        set_param(test_model, "SolverType", "Variable-step");
        set_param(test_model, "MinStep", "0.00001");
        set_param(test_model, "MaxStep", "10");
        set_param(test_model, "InitialStep", "0.001");
        set_param(test_model, "AbsTol", "0.00001");

    end
    
    res_table = table(sim_results.tout, sim_results.y, ...
                      'VariableNames', ["t", "y"]);

    writetable(res_table, solver(i) + "_test_10sec.csv");

    plot(sim_results.tout, sim_results.y), hold on;

end

grid on, title("Simulation results - 10 seconds"), xlabel("t (s)"), ylabel("y"), ...
    legend("ode1", "ode2", "ode4", "ode45");

% Compute errors for 10 second simulations.
errors = [mean(abs(sim_results.y - results(:, 2))), ...
          mean(abs(sim_results.y - results(:, 4))), ...
          mean(abs(sim_results.y - results(:, 6)))];

fprintf(1, "Mean absolute error for ode1: %f\n", errors(1));
fprintf(1, "Mean absolute error for ode2: %f\n", errors(2));
fprintf(1, "Mean absolute error for ode4: %f\n", errors(3));

% Define simulation parameters for 100 second simulations.
stop_time = "100.0";

% Set parameters and run simulations for 100 second simulations.
set_param(test_model, "StopTime", stop_time);
results = zeros(str2double(stop_time) / str2double(fixed_step_size) + 1, ...
                2 * fixed_step_solvers);
figure

for i=1:length(solver)

    set_param(test_model, "Solver", solver(i));
    
    if(fixed(i) == true)
    
        set_param(test_model, "SolverType", "Fixed-step");
        set_param(test_model, "FixedStep", fixed_step_size);
        sim_results = sim(test_model);
    
        results(:, i) = sim_results.tout;
        results(:, i+1) = sim_results.y;

    else

        set_param(test_model, "SolverType", "Variable-step");
        set_param(test_model, "MinStep", "0.00001");
        set_param(test_model, "MaxStep", "10");
        set_param(test_model, "InitialStep", "0.001");
        set_param(test_model, "AbsTol", "0.00001");

    end
    
    res_table = table(sim_results.tout, sim_results.y, ...
                      'VariableNames', ["t", "y"]);

    writetable(res_table, solver(i) + "_test_100sec.csv");

    plot(sim_results.tout, sim_results.y), hold on;

end

grid on, title("Simulation results - 100 seconds"), xlabel("t (s)"), ylabel("y"), ...
    legend("ode1", "ode2", "ode4", "ode45");

% Compute errors for 100 second simulations.
errors = [mean(abs(sim_results.y - results(:, 2))), ...
          mean(abs(sim_results.y - results(:, 4))), ...
          mean(abs(sim_results.y - results(:, 6)))];

fprintf(1, "Mean absolute error for ode1: %f\n", errors(1));
fprintf(1, "Mean absolute error for ode2: %f\n", errors(2));
fprintf(1, "Mean absolute error for ode4: %f\n", errors(3));