from ._variable_step_solver import variable_step_solver
import numpy as np

"""

"""

class RKF(variable_step_solver):

    """
    
    """

    def __init__(self, initial_step_size: float, t0: float = 0, tolerance: float = 0.00001, max_step_size: float = 0.001, min_step_size: float = 0.000001,
                 min_update: float=0.1, max_update: float=4.0, tfinal: float=10.0) -> None:
        
        """
        
        """

        super().__init__(initial_step_size, t0, tolerance, max_step_size, min_step_size, min_update, max_update, tfinal);

        return;

    def _update_step_size(self, q: float) -> None:
        
        """
        
        """
        
        q = np.fmin(self.qmax, np.fmax(self.qmin, q));
        
        self.h = np.fmin(self.tfinal - self.t, np.fmin(self.hmax, np.fmax(self.hmin, q * self.h)));

        return;

    def step(self, model) -> np.ndarray:
        
        """
        
        """

        flag = True;

        while(flag is True):

            x = model.get_state();
            K1 = self.h * model.eval(self.t, x);
            K2 = self.h * model.eval(self.t + 1/4 * self.h, x + 1/4 * K1);
            K3 = self.h * model.eval(self.t + 3/8 * self.h, x + 3/32 * K1 + 9/32 * K2);
            K4 = self.h * model.eval(self.t + 12/13 * self.h, x + 1932/2197 * K1 - 7200/2197 * K2 + 7296/2197 * K3);
            K5 = self.h * model.eval(self.t + self.h, x + 439/126 * K1 - 8 * K2 + 3680/513 * K3 - 845/4104 * K4);
            K6 = self.h * model.eval(self.t + 1/2 * self.h, x - 8/27 * K1 + 2 * K2 - 3544/2565 * K3 + 1859/4104 * K4 - 11/40 * K5);

            R_vec = 1/self.h * np.abs(1/360 * K1 - 128/4275 * K3 - 2197/75240 * K4 + 1/50 * K5 + 2/55 * K6);
            R = np.min(R_vec);
        
            if(R <= self.eps):

                #if(self.h <= self.hmin):

                #    print("The minimum step size is not small enough to guarantee the specified error tolerance.");

                flag = False;
        
            #else:

            q = 0.84 * (self.eps / R)**(1/4);
            print("q: ", q);
            print(q * self.h);
            self._update_step_size(q);

        new_state = x + 25/216 * K1 + 1408/2565 * K3 + 2197/4104 * K4 - 1/5 * K5;
        self._update_time_step();

        return new_state;