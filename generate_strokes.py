import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.stats import norm
from scipy import ndimage
from PIL import Image, ImageDraw

from spline import Spline


def sample_brush(diameter, std=1, color=(1, 1, 1)):
    color = list(color) + [1]
    brush = np.ones((diameter, diameter, 4)) * np.array(color)[None, None, :]

    x = np.linspace(-1 / std, 1 / std, diameter)
    xv, yv = np.meshgrid(x, x)

    normal = norm.pdf(xv, 0, 1) * norm.pdf(yv, 0, 1)

    mask = np.random.random((diameter, diameter))
    mask = ndimage.gaussian_filter(mask, sigma=0.5)
    mask = mask * normal
    mask /= mask.max()

    brush[..., -1] *= mask
    brush = 255 * np.max([np.zeros_like(brush), brush], 0)
    return brush


def draw_point(image, position, color, brush_size=1):
    size = int(48 * brush_size)
    brush = sample_brush(size, 0.3, color).astype(np.uint8)
    brush = Image.fromarray(brush)

    position = np.array(position) - size // 2
    position = position.astype(int)

    image.paste(brush, tuple(position), brush)

    return image


def draw_line(image, start, end, color, start_size=0.4, end_size=0.3):
    start = np.array(start)
    end = np.array(end)

    distance = np.sqrt(np.sum(np.square(end - start)))

    for t in np.linspace(0, 1, int(distance)):
        position = (1 - t) * start + t * end
        size = (1 - t) * start_size + t * end_size

        draw_point(image, position, color, size)


def evaluate_2spline(a, b, c, t):
    """
    Evaluate a piecewise polynomial of n=2.
    """
    x = 2 * t

    center = b * x * (2 - x)
    a = a if x < 1 else c

    return a * x ** 2 - 2 * a * x + a + center


def draw_curve(image, start, control, end, color, sizes=(0.4, 0.6, 0.4)):
    def distance(p, q):
        return np.sqrt(np.sum(np.square(q - p)))

    spline = Spline(start, control, end)
    n = int(distance(start, control) + distance(control, end)) // 2

    for t in np.linspace(0, 1, n):
        position = spline.evaluate(t) + np.random.randn(2)
        # size = evaluate_2spline(*sizes, t)
        size = (1 - t) * sizes[0] + t * sizes[1]

        draw_point(image, position, color, size)


def save_image(image, path, filename):
    if not os.path.isdir(path):
        os.mkdir(path)

    image.save(os.path.join(path, filename))


def float2uint8(value, maximum=1):
    value /= maximum
    return int(256 * value)


def save_label(path, idx, start, control, end, color, sizes):
    start_x, start_y = [float2uint8(value, 255) for value in start]
    ctrl_x, ctrl_y = [float2uint8(value, 255) for value in control]
    end_x, end_y = [float2uint8(value, 255) for value in end]

    red, green, blue = [float2uint8(value) for value in color]
    start_size, end_size = [float2uint8(value) for value in sizes]

    with open(path, 'a') as file:
        file.write(f"{idx};image_{idx:08d}.png;{start_x};{start_y};{ctrl_x};"
                   f"{ctrl_y};{end_x};{end_y};{red};{green};{blue};"
                   f"{start_size};{end_size}\n")


def stroke_generator(n, size):
    for i in tqdm(range(n)):
        image = Image.new('RGB', (size, size), (0, 0, 0))

        color = np.random.random(3)
        start, control, end = np.random.random((3, 2)) * size
        sizes = np.random.random(2) * 0.5 + 0.2

        draw_curve(image, start, control, end, color, sizes)

        yield image, (start, control, end, color, sizes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a brush stroke dataset.")
    parser.add_argument('-n', type=int,
                        help="Number of samples to generate.")
    parser.add_argument('-o', dest='path', type=str, default='images/',
                        help="Directory to save images to.")
    parser.add_argument('--size', dest='size', type=int, default=64,
                        help="Image size.")
    parser.add_argument('-t', dest='test', action='store_true',
                        help="Generate a number of test images without"
                             "saving them.")
    args = parser.parse_args()

    if not args.test:
        with open("labels.csv", 'w') as file:
            file.write("index;image;start_x;start_y;ctrl_x;ctrl_y;end_x;end_y;red;"
                       "green;blue;start_size;end_size\n")

        for idx, (image, params) in enumerate(stroke_generator(args.n, args.size)):
            save_image(image, args.path, f"image_{idx:08d}.png")
            save_label("labels.csv", idx, *params)
    else:
        for idx, (image, params) in enumerate(stroke_generator(25, args.size)):
            plt.subplot(5, 5, idx+1)
            plt.imshow(image)
            plt.axis('off')

        plt.show()
