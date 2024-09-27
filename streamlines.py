import numpy as np
import matplotlib.pyplot as plt
from rk4 import RK4Integrator
from tensorfield import CircularLogTensorField, TensorField, VortexTensorField

class StreamlineIntegrator:
    def __init__(self, tensor_field):
        self.tensor_field = tensor_field
        self.rk4_integrator = RK4Integrator(tensor_field)

    def compute_streamline(self, x0, y0, h=0.1, max_steps=1000):
        """
        Compute a streamline starting from (x0, y0).
        """
        return self.rk4_integrator.integrate(x0, y0, h, max_steps)

    def compute_multiple_streamlines(self, start_points, h=0.1, max_steps=1000):
        """
        Compute multiple streamlines from a list of starting points.
        """
        return [self.compute_streamline(x, y, h, max_steps) for x, y in start_points]

    def visualize(self, show_streamlines=True, show_velocity=True, show_combined=True, n_points=20, h=0.1, max_steps=1000):
        """
        Flexible visualization function for streamlines and velocity field.
        
        :param show_streamlines: Boolean flag to show streamlines
        :param show_velocity: Boolean flag to show velocity field
        :param show_combined: Boolean flag to show combined plot or separate plots
        :param n_points: Number of points for the grid
        :param h: Step size for streamline integration
        :param max_steps: Maximum number of steps for streamline integration
        """
        x = np.linspace(-5, 5, n_points)
        y = np.linspace(-5, 5, n_points)
        X, Y = np.meshgrid(x, y)
        start_points = list(zip(X.flatten(), Y.flatten()))
        
        streamlines = None
        if show_streamlines:
            streamlines = self.compute_multiple_streamlines(start_points, h, max_steps)
        
        U, V = None, None
        if show_velocity:
            U = np.zeros_like(X)
            V = np.zeros_like(Y)
            for i in range(n_points):
                for j in range(n_points):
                    tensor = self.tensor_field(X[i, j], Y[i, j])
                    U[i, j], V[i, j] = tensor[:, 0]
        
        if show_combined:
            plt.figure(figsize=(12, 12))
            if show_streamlines:
                for streamline in streamlines:
                    plt.plot(streamline[:, 0], streamline[:, 1], 'b-', linewidth=0.5, alpha=0.5)
            if show_velocity:
                plt.quiver(X, Y, U, V, color='r', scale=50, width=0.002, alpha=0.7)
            plt.xlim(-5, 5)
            plt.ylim(-5, 5)
            plt.title("2D Tensor Field Visualization")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)
        else:
            if show_streamlines:
                plt.figure(figsize=(10, 10))
                for streamline in streamlines:
                    plt.plot(streamline[:, 0], streamline[:, 1], 'b-', linewidth=0.5, alpha=0.5)
                plt.xlim(-5, 5)
                plt.ylim(-5, 5)
                plt.title("Streamlines of 2D Tensor Field")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.grid(True)
            
            if show_velocity:
                plt.figure(figsize=(10, 10))
                plt.quiver(X, Y, U, V, scale=50, width=0.002)
                plt.xlim(-5, 5)
                plt.ylim(-5, 5)
                plt.title("Velocity Field of 2D Tensor Field")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.grid(True)
        
        plt.show()

# Example usage
tensor_field = CircularLogTensorField()
integrator = StreamlineIntegrator(tensor_field)

# Visualize both streamlines and velocity field as separate figures
integrator.visualize(show_streamlines=True, show_velocity=True, show_combined=False)