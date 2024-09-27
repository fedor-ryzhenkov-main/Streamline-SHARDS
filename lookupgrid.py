import numpy as np

class Cell:
    def __init__(self):
        """
        Represents a grid cell that can contain multiple points.
        """
        self.children = []

    def occupy(self, point):
        """
        Adds a point to the cell.
        """
        self.children.append(point)

    def is_taken(self, x, y, check_callback):
        """
        Checks if the cell is taken based on a callback function that evaluates the distance.
        """
        for p in self.children:
            dx = p[0] - x
            dy = p[1] - y
            dist = np.sqrt(dx * dx + dy * dy)
            if check_callback(dist, p):
                return True
        return False

    def get_min_distance(self, x, y):
        """
        Returns the minimum distance from (x, y) to any point in the cell.
        """
        min_distance = float('inf')
        for p in self.children:
            dx = p[0] - x
            dy = p[1] - y
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < min_distance:
                min_distance = dist
        return min_distance

class LookupGrid:
    def __init__(self, bbox, cell_size):
        """
        Creates a lookup grid based on the bounding box and cell size.
        """
        self.bbox = bbox
        self.cell_size = cell_size
        self.bbox_size = max(bbox['width'], bbox['height'])
        self.cells_count = int(np.ceil(self.bbox_size / cell_size))
        self.cells = {}  # Dictionary of dictionaries for dynamic allocation

    def occupy_coordinates(self, point):
        """
        Occupies the grid cell corresponding to the given point.
        """
        x, y = point
        cell = self.get_cell_by_coordinates(x, y)
        cell.occupy(point)

    def is_taken(self, x, y, check_callback):
        """
        Checks if the point (x, y) is taken by any nearby cell.
        """
        cx, cy = self.grid_coordinates(x, y)
        for col in range(cx - 1, cx + 2):
            if col < 0 or col >= self.cells_count:
                continue
            if col not in self.cells:
                continue
            for row in range(cy - 1, cy + 2):
                if row < 0 or row >= self.cells_count:
                    continue
                if row not in self.cells[col]:
                    continue
                if self.cells[col][row].is_taken(x, y, check_callback):
                    return True
        return False

    def is_outside(self, x, y):
        """
        Checks if the point (x, y) is outside the bounding box.
        """
        return x < self.bbox['left'] or x > self.bbox['left'] + self.bbox['width'] or \
               y < self.bbox['top'] or y > self.bbox['top'] + self.bbox['height']

    def find_nearest(self, x, y):
        """
        Finds the minimum distance from (x, y) to any point in nearby cells.
        """
        cx, cy = self.grid_coordinates(x, y)
        min_distance = float('inf')
        for col in range(cx - 1, cx + 2):
            if col < 0 or col >= self.cells_count:
                continue
            if col not in self.cells:
                continue
            for row in range(cy - 1, cy + 2):
                if row < 0 or row >= self.cells_count:
                    continue
                if row not in self.cells[col]:
                    continue
                d = self.cells[col][row].get_min_distance(x, y)
                if d < min_distance:
                    min_distance = d
        return min_distance

    def get_cell_by_coordinates(self, x, y):
        """
        Retrieves the cell corresponding to the given coordinates, creating it if necessary.
        """
        self.assert_in_bounds(x, y)
        cx, cy = self.grid_coordinates(x, y)
        if cx not in self.cells:
            self.cells[cx] = {}
        if cy not in self.cells[cx]:
            self.cells[cx][cy] = Cell()
        return self.cells[cx][cy]

    def grid_coordinates(self, x, y):
        """
        Converts world coordinates to grid coordinates.
        """
        cx = int(self.cells_count * (x - self.bbox['left']) / self.bbox_size)
        cy = int(self.cells_count * (y - self.bbox['top']) / self.bbox_size)
        return cx, cy

    def assert_in_bounds(self, x, y):
        """
        Asserts that the point (x, y) is within the bounding box.
        """
        if self.is_outside(x, y):
            raise ValueError('Point is out of bounds')
