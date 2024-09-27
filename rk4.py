import numpy as np


class RK4Integrator:
    def __init__(self, tensor_field):
        self.tensor_field = tensor_field

    def step(self, x, y, h):
        """
        Perform one step of RK4 integration.
        """
        k1 = self.tensor_field(x, y)[:, 0]
        k2 = self.tensor_field(x + 0.5*h*k1[0], y + 0.5*h*k1[1])[:, 0]
        k3 = self.tensor_field(x + 0.5*h*k2[0], y + 0.5*h*k2[1])[:, 0]
        k4 = self.tensor_field(x + h*k3[0], y + h*k3[1])[:, 0]
        
        return x + (h/6) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]), \
               y + (h/6) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])

    def integrate(self, x0, y0, h=0.1):
        """
        Compute a streamline starting from (x0, y0).
        """
        streamline = [(x0, y0)]
        x, y = x0, y0
        
        x, y = self.step(x, y, h)
        streamline.append((x, y))
        
        return np.array(streamline)
