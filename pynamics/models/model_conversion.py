#   Copyright 2024 Miguel Loureiro

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""
This module contains helper functions that make pynamics compatible with the Python Control Systems Library.

! ADD PAGE LINKS TO THESE LIBRARIES!!!!!

Functions
---------
pynamics_to_control()
    Convert a pynamics linear state-space model to a control state-space model.

control_to_pynamics()
    Convert a control state-space model to a pynamics state-space model.
"""

from .state_space_models import LinearModel
import control as ct
import numpy as np

def pynamics_to_control(pynamics_model: LinearModel) -> ct.ss:
    """
    Convert a pynamics linear model to a control linear state-space model.

    This method can be used to convert a pynamics model to a control model, thus making the two libraries compatible.

    Parameters
    ----------
    pynamics_model : LinearModel
        A pynamics linear state-space model.

    Returns
    -------
    ct.ss
        A state-space model compatible with the Python Control Systems library.
    """

    return ct.ss(pynamics_model.A, pynamics_model.B, pynamics_model.C, pynamics_model.D);

def control_to_pynamics(control_model: ct.ss, 
                        initial_state: np.ndarray, 
                        initial_control: np.ndarray,
                        input_labels: list[str] | None=None, 
                        output_labels: list[str] | None=None) -> LinearModel:
    """
    Convert a control linear state-space model to a pynamics linear model.

    This method can be used to convert a control model to a pynamics model, thus making the two libraries compatible.

    Parameters
    ----------
    control_model : ct.ss
        A Python Control Systems library linear state-space model.

    initial_state : np.ndarray
        The system's initial state. Should be an array shaped (n, 1), where
        n is the number of variables.

    initial_control : np.ndarray
        The inputs' initial value(s). Should be an array shaped (u, 1), where
        u is the number of input variables.

    input_labels : list[str] | None, optional
        Names of the input variables. Should be a list of length n.

    output_labels : list[str] | None, optional
        Names of the output variables. Should be a list of length u.

    Returns
    -------
    LinearModel
        A pynamics linear model.
    """

    return LinearModel(initial_state, 
                       initial_control, 
                       control_model.A, 
                       control_model.B, 
                       control_model.C, 
                       control_model.D, 
                       input_labels, 
                       output_labels);