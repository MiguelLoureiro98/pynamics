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

"""
This module containts the abstract 'fixed step solver' class from which all fixed step solvers inherit / derive.
"""

class _FixedStepSolver(ABC):

    """
    Generic fixed step solver class to serve as [the] mother class to all fixed step solvers supported by this package.

    Attributes
    ----------------------------------------------------------------------------------
    h: float
    Solver step size.

    t: float
    The current solver time step.

    Methods
    ----------------------------------------------------------------------------------
    __init__
    _update_time_step
    get_time_step
    step
    """

    def __init__(self, step_size: float, t0: float=0.0) -> None:

        """
        
        """

        super().__init__();
        self.h = step_size;
        self.t = t0;

        return;

    def _update_time_step(self) -> None:

        """
        
        """

        self.t += self.h;

        return; 

    def get_time_step(self) -> None:

        """
        
        """

        return self.t;

    @abstractmethod
    def step(self) -> tuple:

        """
        
        """

        pass