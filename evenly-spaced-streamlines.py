import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from lookupgrid import LookupGrid
from tensorfield import CircularLogTensorField
from rk4 import RK4Integrator
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


FORWARD = 1
BACKWARD = 2
DONE = 3

class StreamlineIntegrator:
    def __init__(self, start, grid, tensor_field, config):
        """
        Initializes the streamline integrator with a starting point, grid, tensor field, and configuration.
        """
        self.points = [np.array(start, dtype=np.float64)]
        self.pos = np.array(start, dtype=np.float64)
        self.state = FORWARD
        self.candidate = None
        self.last_checked_seed = -1
        self.own_grid = LookupGrid(config['boundingBox'], config['timeStep'] * 0.9)
        self.grid = grid
        self.config = config
        self.start = np.array(start, dtype=np.float64)
        self.tensor_field = tensor_field
        self.h = config['timeStep']

    def get_streamline(self):
        """
        Returns the list of points in the streamline.
        """
        return self.points

    def get_next_valid_seed(self):
        while self.last_checked_seed < len(self.points) - 1:
            self.last_checked_seed += 1
            p = self.points[self.last_checked_seed]
            v = self.normalized_vector_field(p)
            if v is None:
                continue

            # First side
            cx = p[0] - v[1] * self.config['dSep']
            cy = p[1] + v[0] * self.config['dSep']
            if not self.grid.is_outside(cx, cy) and not self.grid.is_taken(cx, cy, self.check_dsep):
                logging.debug(f"Valid seed found at ({cx}, {cy})")
                return np.array([cx, cy], dtype=np.float64)

            # Other side
            ox = p[0] + v[1] * self.config['dSep']
            oy = p[1] - v[0] * self.config['dSep']
            if not self.grid.is_outside(ox, oy) and not self.grid.is_taken(ox, oy, self.check_dsep):
                logging.debug(f"Valid seed found at ({ox}, {oy})")
                return np.array([ox, oy], dtype=np.float64)

            logging.debug(f"No valid seed at point {p}")
        logging.debug("No more seeds found along this streamline")
        return None

    def check_dtest(self, distance_to_candidate, p=None):
        """
        Checks if the distance to a candidate point is less than dTest.
        """
        if self.is_same(distance_to_candidate, self.config['dTest']):
            return False
        return distance_to_candidate < self.config['dTest']

    def check_dsep(self, distance_to_candidate, p=None):
        """
        Checks if the distance to a candidate point is less than dSep.
        """
        if self.is_same(distance_to_candidate, self.config['dSep']):
            return False
        return distance_to_candidate < self.config['dSep']

    def is_same(self, a, b):
        """
        Checks if two floating point numbers are approximately the same, avoiding floating point errors.
        """
        return abs(a - b) < 1e-4

    def next(self):
        """
        Advances the streamline by one step, growing it in the current direction.
        """
        while True:
            self.candidate = None
            if self.state == FORWARD:
                point = self.grow_forward()
                if point is not None:
                    self.points.append(point)
                    self.own_grid.occupy_coordinates(point)
                    self.pos = point
                    should_pause = self.notify_point_added(point)
                    if should_pause:
                        return False  # Pause to allow for new streamline
                else:
                    if self.config.get('forwardOnly', False):
                        self.state = DONE
                    else:
                        self.pos = self.start
                        self.state = BACKWARD
            elif self.state == BACKWARD:
                point = self.grow_backward()
                if point is not None:
                    self.points.insert(0, point)
                    self.pos = point
                    self.own_grid.occupy_coordinates(point)
                    should_pause = self.notify_point_added(point)
                    if should_pause:
                        return False  # Pause to allow for new streamline
                else:
                    self.state = DONE
            if self.state == DONE:
                for p in self.points:
                    self.occupy_point_in_grid(p)
                return True  # Streamline is complete

    def occupy_point_in_grid(self, p):
        """
        Marks a point as occupied in the main grid.
        """
        self.grid.occupy_coordinates(p)

    def grow_forward(self):
        """
        Grows the streamline forward by one RK4 step.
        """
        displacement = self.rk4_step(self.pos, self.h)
        if displacement is None:
            return None  # Hit a singularity
        return self.grow_by_velocity(self.pos, displacement)

    def grow_backward(self):
        """
        Grows the streamline backward by one RK4 step.
        """
        displacement = self.rk4_step(self.pos, self.h)
        if displacement is None:
            return None
        displacement = -displacement
        return self.grow_by_velocity(self.pos, displacement)

    def grow_by_velocity(self, pos, displacement):
        """
        Attempts to grow the streamline by the given displacement, checking termination conditions.
        """
        candidate = pos + displacement
        if self.grid.is_outside(candidate[0], candidate[1]):
            return None
        if self.grid.is_taken(candidate[0], candidate[1], self.check_dtest):
            return None
        if self.own_grid.is_taken(candidate[0], candidate[1], self.time_step_check):
            return None
        return candidate

    def time_step_check(self, distance_to_candidate, p=None):
        """
        Checks if the distance to a candidate point is less than the time step, indicating a possible loop.
        """
        return distance_to_candidate < self.config['timeStep'] * 0.9

    def notify_point_added(self, point):
        """
        Callback function executed when a point is added to the streamline.
        """
        should_pause = False
        if self.config.get('onPointAdded', None) is not None:
            previous_point = self.points[-2] if self.state == FORWARD and len(self.points) >= 2 else None
            should_pause = self.config['onPointAdded'](point, previous_point, self.config, self.points)
        return should_pause

    def normalized_vector_field(self, p):
        """
        Evaluates and normalizes the vector field at point p.
        """
        v = self.config['vectorField'](p, self.points, self.state == DONE)
        if v is None or np.isnan(v[0]) or np.isnan(v[1]):
            return None
        norm = np.linalg.norm(v)
        if norm == 0:
            return None
        return v / norm

    def rk4_step(self, pos, h):
        """
        Performs a single RK4 integration step to compute the displacement.
        """
        v1 = self.normalized_vector_field(pos)
        if v1 is None:
            return None
        k1 = h * v1

        v2_pos = pos + 0.5 * k1
        v2 = self.normalized_vector_field(v2_pos)
        if v2 is None:
            return None
        k2 = h * v2

        v3_pos = pos + 0.5 * k2
        v3 = self.normalized_vector_field(v3_pos)
        if v3 is None:
            return None
        k3 = h * v3

        v4_pos = pos + k3
        v4 = self.normalized_vector_field(v4_pos)
        if v4 is None:
            return None
        k4 = h * v4

        displacement = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        return displacement

class EvenlySpacedStreamlines:
    def __init__(self, tensor_field, dsep, dtest):
        """
        Initializes the EvenlySpacedStreamlines class with the given tensor field, separation distances, and plot setup.
        """
        logging.info("Initializing EvenlySpacedStreamlines")
        self.tensor_field = tensor_field
        self.dsep = dsep
        self.dtest = dtest

        self.streamlines = []
        self.lookup_grid = LookupGrid(
            bbox={'left': -5, 'top': -5, 'width': 10, 'height': 10},
            cell_size=dsep
        )

        self.config = {
            'boundingBox': {'left': -5, 'top': -5, 'width': 10, 'height': 10},
            'dSep': dsep,
            'dTest': dtest,
            'timeStep': 0.1,
            'vectorField': self.vector_field,
            'onPointAdded': self.on_point_added,
            'forwardOnly': False,
            'seedArray': [],
        }
        self.integrators = []

        # Set up the plot
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_title("Evenly Spaced Streamlines (Real-time)")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.grid(True)
        logging.info("Plot setup complete")

    def vector_field(self, p, points, done):
        """
        Returns the vector at point p from the tensor field.
        """
        x, y = p[0], p[1]
        v = self.tensor_field.get_vector(x, y)
        return v

    def on_point_added(self, point, previous_point, config, points):
        """
        Callback executed when a point is added to a streamline. Updates the plot in real-time.
        """
        if previous_point is not None:
            self.ax.plot(
                [previous_point[0], point[0]],
                [previous_point[1], point[1]],
                'b-',
                linewidth=0.5
            )
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        return False  # Continue integration without pausing

    def compute_streamlines(self, x_range, y_range):
        logging.info("Starting streamline computation")
        start_time = time.time()

        # Initialize the first streamline with a random seed
        seed = self.get_random_seed(x_range, y_range)
        self.add_streamline(seed)
        logging.debug(f"Initial seed added: {seed}")

        while self.integrators:
            integrator = self.integrators[-1]
            logging.debug(f"Processing streamline with seed {integrator.start}")
            completed = integrator.next()
            if completed:
                # Streamline is complete; finalize it
                streamline = integrator.get_streamline()
                self.streamlines.append(np.array(streamline))
                logging.info(f"Streamline with seed {integrator.start} completed")
                # After completion, try to find the next valid seed
                new_seed = integrator.get_next_valid_seed()
                if new_seed is not None:
                    logging.info(f"New seed found after completing streamline: {new_seed}")
                    self.add_streamline(new_seed)
                else:
                    # No more seeds to explore; pop the streamline
                    self.integrators.pop()
                    logging.debug(f"Streamline with seed {integrator.start} popped from stack")
            else:
                # Integration paused to start a new streamline
                logging.debug(f"Integration paused for streamline with seed {integrator.start}")
                # Continue processing without popping
                pass

        end_time = time.time()
        logging.info(f"Streamline computation complete. Total streamlines: {len(self.streamlines)}")
        logging.info(f"Time taken: {end_time - start_time:.2f} seconds")

    def add_streamline(self, seed):
        """
        Adds a new streamline starting from the given seed.
        """
        logging.info(f"Adding streamline with seed {seed}")
        integrator = StreamlineIntegrator(seed, self.lookup_grid, self.tensor_field, self.config)
        self.integrators.append(integrator)

    def get_random_seed(self, x_range, y_range):
        """
        Generates a random seed within the specified x and y ranges.
        """
        x = np.random.uniform(*x_range)
        y = np.random.uniform(*y_range)
        return np.array([x, y], dtype=np.float64)

    def finalize_plot(self):
        """
        Finalizes the plot by turning off interactive mode and displaying the plot.
        """
        logging.info("Finalizing plot")
        plt.ioff()
        plt.show()

# Example usage
if __name__ == "__main__":
    logging.info("Starting streamline generation process")
    # You should define or import your tensor field class here
    tensor_field = CircularLogTensorField()
    streamline_generator = EvenlySpacedStreamlines(tensor_field, dsep=0.1, dtest=0.005)
    streamline_generator.compute_streamlines(x_range=(-5, 5), y_range=(-5, 5))
    streamline_generator.finalize_plot()
    logging.info("Streamline generation process complete")