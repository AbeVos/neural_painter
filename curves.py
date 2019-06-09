import numpy as np


class Bezier():
    def __init__(self, start, control, end):
        """
        Create a Bezier curve from three control points.
        """
        self.start = np.array(start)
        self.control = np.array(control)
        self.end = np.array(end)

        self.length = self._compute_length()

    def evaluate(self, t):
        return self.control + (1 - t) ** 2 * (self.start - self.control) \
            + t ** 2 * (self.end - self.control)

    def _compute_length(self):
        a = self.start
        distance = 0

        for t in np.linspace(0, 1, 10)[1:]:
            b = self.evaluate(t)
            distance += np.linalg.norm(a - b)
            a = b

        return distance
