# Welcome to the Pynamics documentation

Pynamics is a simple, lightweight Python package to simulate dynamical systems.

It is mainly inteded as a support package for more advanced control design projects, especially projects seeking to implement advanced control systems leveraging the predictive capabilities of machine learning algorithms.

The package provides classes to model a system (linear and nonlinear state-space models) and a simulator class that can be used to run different types of simulations. Limited control capabilities are also provided, namely a controller base class that users can build upon to design their own controllers.

## Main features
- **Simulations**: simulate dynamical systems in Python using our simulator class. Only fixed-step solvers are supported at the moment.

- **Plot results**: plot the simulation results automatically.

- **State-space models**: model your system using our generic linear and nonlinear state-space models.

## Documentation Guide
See our [installation](installation.md) guide for detailed instructions on how to install Pynamics.

If you require more detailed information on how to use the package (including examples), check out the [API reference](API_ref/index.md).

For more information on the package's author and development process, see the [About](about.md) section.