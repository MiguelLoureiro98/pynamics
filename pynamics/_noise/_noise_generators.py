import numpy as np

"""
This module contains helper functions to help generate noise.

As of now, only white noise generation is supported.
However, support for coloured noise may come in the future.
"""

def white_noise(n_series: int, n_samples: int, power: int | float, seed: int) -> np.ndarray:
    
    """
    Generate White Gaussian Noise.

    This function generates one or more time series of white gaussian noise, each with a length of n_samples.

    Parameters
    ----------
    n_series : int
        _description_

    n_samples : int
        _description_

    power : int | float
        White noise power in Watt.

    seed : int
        Random seed to allow for ... .

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