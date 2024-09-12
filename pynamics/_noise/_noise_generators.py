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
This module contains helper functions to help generate noise.

As of now, only white noise generation is supported.
However, support for coloured noise may come in the future.
"""

import numpy as np

def _white_noise(n_series: int, n_samples: int, power: int | float, seed: int) -> np.ndarray:
    
    """
    Generate White Gaussian Noise.

    This function generates one or more time series of white gaussian noise, each with a length of n_samples.

    Parameters
    ----------
    n_series : int
        Number of white noise time series to generate.

    n_samples : int
        Number of samples (i.e. length) of each time series.

    power : int | float
        White noise power in Watt.

    seed : int
        Random seed to allow for reproducible white noise generation.

    Returns
    -------
    np.ndarray
        An array shaped (n_series, n_samples) containing the generated white noise time series.

    Raises
    ------
    TypeError
        If the noise power is not an integer or a float.

    ValueError
        If the noise power is negative.

    TypeError
        If the seed is not an integer.
    """

    if(isinstance(power, int) is False and isinstance(power, float) is False):

        raise TypeError("The noise power must be either an integer or a float.");

    if(power < 0):

        raise ValueError("The noise power must be positive.");

    if(isinstance(seed, int) is False):

        raise TypeError("The seed must be an integer.");

    if(n_series == 1):

        noise = np.random.default_rng(seed=seed).normal(scale=np.sqrt(power), size=(1, n_samples));
    
    else:

        noise = np.zeros(shape=(n_series, n_samples));
    
        for ind in range(n_series):

            noise[ind, :] = np.random.default_rng(seed=seed).normal(scale=np.sqrt(power), size=(1, n_samples));

    return noise;