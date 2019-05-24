import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.stats import norm
from PIL import Image

from spline import Spline


def brush_gaussian(diameter):
    """
    Create a Gaussian alphamap of a given diameter.
    The Gaussian's standard deviation is set to fit the curve from the 0.001-th
    to the 0.999-th percentile within `diameter`.
    """
    diameter = int(round(diameter))
    x = np.linspace(norm.ppf(0.001), norm.ppf(0.999), diameter)

    xx, yy = np.meshgrid(x, x)
    normal = norm.pdf(xx, 0, 1) * norm.pdf(yy, 0, 1)

    normal /= normal.max()

    return normal


def draw_alpha(dst, src, position):
    """
    Draw a small alphamap onto a larger one.
    The source alphamap can be offset if.
    """
    x, y = position
    h, w = src.shape
    H, W = dst.shape

    x -= w // 2
    y -= h // 2

    # Correct src if it draws outside the dst borders.
    if y < 0:
        src = src[abs(y):]
        y = 0
    if x < 0:
        src = src[:, abs(x):]
        x = 0

    h, w = src.shape

    if H - (h + y) < 0:
        src = src[:H - (h + y)]
    if W - (w + x) < 0:
        src = src[:, :W - (w + x)]

    # Blend the source image with the patch of the destination image.
    patch = dst[y:y+h, x:x+w]
    patch = src + (1 - src) * patch

    dst[y:y+h, x:x+w] = patch


def draw_curve(dst, start, control, end, size_start, size_end,
               brush=brush_gaussian):
    """
    Draw a spline curve unto a canvas.
    """
    def distance(p, q):
        """Euclidian distance between two points."""
        return np.sqrt(np.sum(np.square(np.subtract(q, p))))

    # Resize values to canvas size.
    dst_size = dst.shape[0]

    start = np.array(start) * dst_size
    control = np.array(control) * dst_size
    end = np.array(end) * dst_size
    size_start *= dst_size
    size_end *= dst_size

    spline = Spline(start, control, end)
    n = int(distance(start, control) + distance(control, end))

    for t in np.linspace(0, 1, n):
        size = (1 - t) * size_start + t * size_end

        noise = size / (0.5 * dst_size) * np.random.rand(2)
        x, y = spline.evaluate(t) + noise

        draw_alpha(dst, brush(size), (int(round(x)), int(round(y))))


def generate_parameters():
    start, control, end = np.random.rand(3, 2)
    color = np.random.rand(3)
    start_size, end_size = np.random.random(2) * 0.25 + 0.05

    return (start, control, end), color, (start_size, end_size)


def stroke_generator(n, size=64):
    for idx in range(n):
        # Create random stroke parameters.
        positions, color, sizes = generate_parameters()

        # Initialize a canvas with `color`.
        canvas = np.ones((size, size, 3))
        canvas *= color

        # Initialize an empty alphamap.
        alphamap = np.zeros((size, size))

        # Draw the curve onto the alphamap.
        draw_curve(alphamap, *positions, *sizes)

        # Combine the colored canvas and the stroke's alphamap.
        canvas = np.concatenate((canvas, alphamap[..., None]), -1)

        yield canvas, (positions, color, sizes)


def generate_strokes(args):
    label_path = os.path.join(args.output_root, 'labels.csv')
    image_root = os.path.join(args.output_root, 'images')

    os.makedirs(image_root, exist_ok=True)
    label = "{};{};{};{};{};{};{};{};{};{};{};{};{}\n"
    header = label.format('index', 'image', 'start_x', 'start_y', 'ctrl_x',
                          'ctrl_y', 'end_x', 'end_y', 'red', 'green', 'blue',
                          'size_start', 'size_end')

    with open(label_path, 'w') as file:
        file.write(header)

        for idx, (stroke, parameters) in tqdm(enumerate(
                stroke_generator(args.number, args.size)), total=args.number): 
            image_name = f"stroke_{idx:08d}.png"
            image_path = os.path.join(image_root, image_name)

            stroke = (255 * stroke).astype(np.uint8)
            image = Image.fromarray(stroke)

            positions, color, sizes = parameters
            positions = (255 * np.array(positions)).astype(np.uint8).ravel()
            color = (255 * np.array(color)).astype(np.uint8)
            sizes = (255 * np.array(sizes)).astype(np.uint8)

            row = label.format(idx, image_name, *positions, *color, *sizes)
            file.write(row)

            image.save(image_path)


def test_strokes():
    """
    Generate 25 test strokes and plot them.
    """
    for idx, (stroke, _) in tqdm(enumerate(stroke_generator(25)), total=25):
        plt.subplot(5, 5, idx+1)
        plt.imshow(stroke)
        plt.axis('off')

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a stroke dataset.")
    parser.add_argument('-n', dest='number', type=int, default=100,
                        help="Number of images to generate.")
    parser.add_argument('-s', dest='size', type=int, default=64,
                        help="Size of the images.")
    parser.add_argument('-o', dest='output_root', type=str, default='strokes/',
                        help="DIrectory to write the data to.")
    parser.add_argument('-t', dest='test', action='store_true',
                        help="Draw a number of test strokes and exit.")
    args = parser.parse_args()

    if args.test:
        test_strokes()
    else:
        generate_strokes(args)
