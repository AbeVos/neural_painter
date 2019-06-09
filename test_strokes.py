import numpy as np
import matplotlib.pyplot as plt

from generate_strokes import draw_curve

def draw_stroke(dst, action, color, offset=(0, 0)):
    height, width = dst.shape[:2]

    start_x, start_y, ctrl_x, ctrl_y, end_x, end_y = action[:6]
    start_size, end_size = action[6:]

    alphamap = np.zeros((height, width))
    draw_curve(alphamap, (start_x, start_y), (ctrl_x, ctrl_y), (end_x, end_y),
               start_size, end_size)




if __name__ == "__main__":
    action = np.random.rand(8)
    canvas = np.ones((128, 128, 3))

    draw_stroke(canvas, action, np.array([0.5, 0.2, 0.7]))

    plt.imshow(canvas)
    plt.show()
