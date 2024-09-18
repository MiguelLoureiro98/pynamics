# Welcome to Pynamics

Pynamics is a simple, lightweight Python package to simulate dynamical systems.

It is mainly inteded as a support package for more advanced control design projects, especially projects seeking to implement advanced control systems leveraging the predictive capabilities of machine learning algorithms.

The package provides classes to model a system (linear and nonlinear state-space models) and a simulator class that can be used to run different types of simulations. Limited control capabilities are also provided, namely a controller base class that users can build upon to design their own controllers.

Please note that this package is NOT appropriate for extensive, extremely precise simulations.

# Main features
- **Simulations**: simulate dynamical systems in Python using our simulator class. Only fixed-step solvers are supported at the moment.

- **Plot results**: plot the simulation results automatically.

- **State-space models**: model your system using our generic linear and nonlinear state-space models.

# Installation

## Using pip
The easiest way to install [Pynamics](index.md) is to install it directly from PyPi using pip:

```
pip install pynamics
```

This will install the package along with its dependencies.

If you wish to contribute to the package, install its development dependencies by typing:

```
pip install "pynamics[dev]"
```

## Using git
Alternatively, the package can be installed via git. To do so, you must first clone the repository:

```
git clone https://github.com/MiguelLoureiro98/pynamics.git
```

Once the repository has been cloned, move into its directory and install the package using pip:

```
cd pynamics
pip install .
```

# Documentation

Check out the official documentation: https://miguelloureiro98.github.io/pynamics/

# About

This package was developed by [Miguel Loureiro](https://www.linkedin.com/in/miguel-santos-loureiro/), a Mechanical Engineer who specialises in control systems, machine learning and optimisation algorithms.

If you're interested in any of these, feel free to contact me via email: miguel.santos.loureiro@gmail.com