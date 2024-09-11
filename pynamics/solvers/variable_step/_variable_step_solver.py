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

from abc import ABC, abstractmethod
import numpy as np

"""
This file containts the abstract 'variable step solver' class from which all variable step solvers inherit / are derived.
"""

class variable_step_solver(ABC):

    """
    Generic variable step solver class to serve as [the] mother class to all variable step solvers supported by this package.
    """

    def __init__(self, initial_step_size: float, t0: float=0.0, tolerance: float=1e-5, max_step_size: float=1e-3, min_step_size: float=1e-6, 
                 min_update: float=0.1, max_update: float=4.0, tfinal: float=10.0) -> None:

        """
        
        """

        super().__init__();
        self.h = initial_step_size;
        self.hmin = min_step_size;
        self.hmax = max_step_size;
        self.qmin = min_update;
        self.qmax = max_update;
        self.t = t0;
        self.tfinal = tfinal;
        self.eps = tolerance;

        return;

    def _update_time_step(self) -> None:

        """
        
        """

        self.t += self.h;

        return; 

    @abstractmethod
    def _update_step_size(self) -> None:

        """
        
        """

        pass

    def get_time_step(self) -> None:

        """
        
        """

        return self.t;

    def get_step_size(self) -> None:

        """
        
        """

        return self.h;

    @abstractmethod
    def step(self) -> np.ndarray:

        """
        
        """

        pass