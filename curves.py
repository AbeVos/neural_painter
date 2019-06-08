import numpy as np


class Bezier():
    def __init__(self, start, control, end):
        self.start = np.array(start)
        self.control = np.array(control)
        self.end = np.array(end)

        self.delta = 0.01

    def evaluate(self, t):
        '''
        t -= self.delta / 2
        s = t + self.delta
        first_start = (1 - t) * self.start + t * self.control
        first_end = (1 - t) * self.control + t * self.end

        second_start = (1 - s) * self.start + s * self.control
        second_end = (1 - s) * self.control + s * self.end

        first_dir = (first_end[1] - first_start[1]) \
            / (first_end[0] - first_start[0])
        second_dir = (second_end[1] - second_start[1]) \
            / (second_end[0] - second_start[0])

        x = (second_start[1] - second_dir * second_start[0]
             - first_start[1] + first_dir * first_start[0]) \
            / (first_dir - second_dir)
        y = first_dir * (x - first_start[0]) + first_start[1]

        return np.array([x, y])
        '''

        return self.control + (1 - t) ** 2 * (self.start - self.control) \
            + t ** 2 * (self.end - self.control)
