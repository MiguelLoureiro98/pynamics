# Models subpackage

The Pynamics models subpackage implements generic state-space models that can be used to describe the systems one wishes to simulate.
Additionally, it provides helpful model conversion functions.

## Modules
- [Base](base.md): Contains the model base class.

- [State-space models](state_space/index.md): Implements linear and nonlinear state-space models.

- [Model conversions](conversions.md): Provides functions to convert [pynamics](../../index.md) models into [control](https://github.com/python-control/python-control) models and vice-versa.