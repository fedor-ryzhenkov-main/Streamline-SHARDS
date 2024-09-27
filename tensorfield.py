import numpy as np


class TensorField:
    def __init__(self):
        """
        Initialize the TensorField class.
        """
        pass

    def __call__(self, x, y):
        """
        Define the tensor field here.
        Returns a 2x2 matrix for each (x, y) point.
        """
        # Example tensor field: rotation + scaling
        theta = 0.5 * np.arctan2(y, x)
        scale = np.sqrt(x**2 + y**2)
        return np.array([
            [scale * np.cos(theta), -scale * np.sin(theta)],
            [scale * np.sin(theta), scale * np.cos(theta)]
        ])
    
    def get_vector(self, x, y):
        """
        Get the vector field for a given (x, y) point.
        Returns a 2-element array [vx, vy].
        """
        tensor = self(x, y)
        return tensor[:, 0]


class VortexTensorField(TensorField):
    def __init__(self, strength=1.0, center=(0, 0)):
        """
        Initialize the VortexTensorField class.
        
        :param strength: The strength of the vortex. Positive values create
                         counterclockwise rotation, negative values create
                         clockwise rotation.
        :param center: The (x, y) coordinates of the vortex center.
        """
        super().__init__()
        self.strength = strength
        self.center = np.array(center)

    def __call__(self, x, y):
        """
        Define the vortex tensor field.
        Returns a 2x2 matrix for each (x, y) point.
        """
        r = np.array([x, y]) - self.center
        r_squared = np.sum(r**2)
        
        # Add a small epsilon to avoid division by zero
        epsilon = 1e-10
        r_squared += epsilon
        
        factor = self.strength / (2 * np.pi * r_squared)
        return factor * np.array([
            [-r[1], -r[0]],
            [r[0], -r[1]]
        ])



class CircularLogTensorField(TensorField):
    def __init__(self):
        """
        Initialize the CircularLogTensorField class.
        """
        super().__init__()

    def __call__(self, x, y):
        """
        Define the circular-logarithmic tensor field.
        Returns a 2x2 matrix for each (x, y) point.
        """
        l = np.sqrt(x**2 + y**2)
        
        # Avoid log(0) by adding a small epsilon
        epsilon = 1e-10
        abs_x = np.abs(x) + epsilon
        
        # Calculate velocity components
        vx = np.cos(l)
        vy = np.log(abs_x)
        
        # Construct the tensor field
        return np.array([
            [vx, -vy],
            [vy, vx]
        ])

