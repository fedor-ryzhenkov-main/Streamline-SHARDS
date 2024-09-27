import logging
import numpy as np


class Cell:
    def __init__(self):
        self.children = []

    def occupy(self, point):
        self.children.append(point)

    def is_taken(self, x, y, check_callback):
        for p in self.children:
            dx = p[0] - x
            dy = p[1] - y
            dist = np.sqrt(dx * dx + dy * dy)
            if check_callback(dist, p):
                return True
        return False

    def get_min_distance(self, x, y):
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
        self.bbox = bbox
        self.cell_size = cell_size
        self.bbox_size = max(bbox['width'], bbox['height'])
        self.cells_count = int(np.ceil(self.bbox_size / cell_size))
        self.cells = {}
        logging.debug(f"LookupGrid initialized with bbox {bbox} and cell_size {cell_size}")

    def occupy_coordinates(self, point):
        x, y = point
        cell = self.get_cell_by_coordinates(x, y)
        cell.occupy(point)
        logging.debug(f"Point {point} occupied in grid")

    def is_taken(self, x, y, check_callback):
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
        outside = x < self.bbox['left'] or x > self.bbox['left'] + self.bbox['width'] or \
                  y < self.bbox['top'] or y > self.bbox['top'] + self.bbox['height']
        if outside:
            logging.debug(f"Point ({x}, {y}) is outside the bounding box")
        return outside

    def find_nearest(self, x, y):
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
        logging.debug(f"Nearest distance to ({x}, {y}) is {min_distance}")
        return min_distance

    def get_cell_by_coordinates(self, x, y):
        self.assert_in_bounds(x, y)
        cx, cy = self.grid_coordinates(x, y)
        if cx not in self.cells:
            self.cells[cx] = {}
        if cy not in self.cells[cx]:
            self.cells[cx][cy] = Cell()
        return self.cells[cx][cy]

    def grid_coordinates(self, x, y):
        cx = int(self.cells_count * (x - self.bbox['left']) / self.bbox_size)
        cy = int(self.cells_count * (y - self.bbox['top']) / self.bbox_size)
        return cx, cy

    def assert_in_bounds(self, x, y):
        if self.is_outside(x, y):
            raise ValueError('Point is out of bounds')