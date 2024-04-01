import numpy as np

"""
This module contains helper functions to help generate noise.

As of now, only white noise generation is supported.
However, support for coloured noise may come in the future.
"""

def white_noise(n_series: int, n_samples: int, power: int | float, seed: int) -> np.ndarray:
    
    """
    _summary_

    _extended_summary_

    Parameters
    ----------
    n_series : int
        _description_

    n_samples : int
        _description_

    power : int | float
        _description_

    seed : int
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """

    if(n_series == 1):

        noise = np.random.default_rng(seed=seed).normal(scale=np.sqrt(power), size=(1, n_samples));
    
    else:

        noise = np.zeros(shape=(n_series, n_samples));
    
        for ind in range(n_series):

            noise[ind, :] = np.random.default_rng(seed=seed).normal(scale=np.sqrt(power), size=(1, n_samples));

    return noise;