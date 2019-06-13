import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from scipy.stats import norm

from curves import Bezier

STENCIL_SIZE = 64


class Brush():
    def __init__(self, brush, stencil_size=STENCIL_SIZE, n_per_size=10):
        self.min_size = stencil_size // 10
        self.max_size = stencil_size // 2
        self.stencil_size = stencil_size

        self._brush_dict = self._create_brush_dict(
            brush, self.min_size, self.max_size, n_per_size)

    def _create_brush_dict(self, brush, min_size, max_size, n):
        """
        Generate a number of brush alpha's for different brush sizes.
        """
        brush_dict = {}
        for size in range(min_size, max_size + 1):
            brush_dict[size] = [brush(size) for _ in range(n)]

        return brush_dict

    def show_brush_dict(self, n=5):
        if n is 0:
            n = len(self._brush_dict[self.min_size])

        canvas = np.zeros(
            (self.stencil_size * len(self._brush_dict),
             self.stencil_size * n))

        for idx, size in enumerate(range(self.min_size, self.max_size + 1)):
            for jdx in range(n):
                alpha = self._brush_dict[size][jdx]
                patch = np.zeros((self.stencil_size, self.stencil_size))
                draw_alpha(
                    patch, alpha, (self.stencil_size // 2, self.stencil_size // 2))

                canvas[self.stencil_size*idx:self.stencil_size*(idx+1),
                       self.stencil_size*jdx:self.stencil_size*(jdx+1)] = patch

        plt.imshow(canvas)
        plt.axis('off')
        plt.show()

    def draw_stroke(self, canvas, action, color, offset=(0, 0)):
        """
        Draw a stroke onto a canvas.
        A stroke's shape consists of a start, control, and end position, and
        a start and end size. The stroke shape is combined with a color to
        create the complete stroke.

        The stroke image in blended onto a canvas, which can be larger than
        the stroke image itself.

        Parameters
        ----------
        canvas
        action : list of float
            A list of values describing the stroke's control points and
            sizes.
            Action values lie in the range [0, 1].
        color : tuple of float,
        offset : tuple of float
        """
        positions = (self.stencil_size * action[:6]).reshape(-1, 2) \
            + np.array(offset)[None, :]
        start, ctrl, end = positions
        start_size, end_size = action[6:]

        curve = Bezier(start, ctrl, end)
        alphamap = np.zeros_like(canvas[..., 0])

        # Draw alphamap.
        for t in np.linspace(0, 1, int(curve.length)):
            position = curve.evaluate(t).astype(int)
            size = (1 - t) * start_size + t * end_size
            size = (1 - size) * self.min_size + size * self.max_size
            size = int(round(size))

            alpha = random.choice(self._brush_dict[size])

            draw_alpha(alphamap, alpha, position)

        # Draw color to the canvas based on the generated alphamap.
        canvas = np.copyto(canvas,
            alphamap[..., None] * color[None, None, :] \
            + (1 - alphamap[..., None]) * canvas)


def transform(image, A, resolution=64):
    """
    Transform an image using a sample field.

    A grid of sampling points is placed over the image uniformly (the
    number of points is determined by `resolution`).
    These points are transformed using the matrix `A` after which the
    average value under each point is sampled.
    These sampled are then reshaped into the transformed image.
    """
    height, width = image.shape

    # Create the sample field.
    x = np.linspace(-0.5, 0.5, resolution)
    xx, yy = np.meshgrid(x, x)
    sample_field = np.stack((xx, yy)).reshape(2, -1)
    sample_field += 0.005 * np.random.randn(*sample_field.shape)

    # Transform the sample coordinates.
    A = np.linalg.inv(A)
    sample_field = A @ sample_field

    sample_field += 0.5
    sample_field = np.clip(sample_field, 0, 1)

    sample_field *= height - 1

    # Sample the image.
    btm = np.floor(sample_field).astype(int)
    top = np.ceil(sample_field).astype(int)

    sample = np.stack((
        image[top[1], btm[0]],
        image[btm[1], btm[0]],
        image[top[1], top[0]],
        image[btm[1], top[0]])).mean(0)

    sample = sample.reshape(resolution, resolution)

    return sample


def brush_gaussian(diameter):
    """
    Create a Gaussian alphamap of a given diameter.
    The Gaussian's standard deviation is set to fit the curve from the 0.001-th
    to the 0.999-th percentile within `diameter`.
    """
    diameter = int(round(diameter))
    x = np.linspace(norm.ppf(0.001), norm.ppf(0.999), 2 * diameter)

    xx, yy = np.meshgrid(x, x)
    normal = norm.pdf(xx, 0, 0.5) * norm.pdf(yy, 0, 0.5)

    return normal


def brush_paint(diameter):
    """
    Sample from a paint brush.

    The paint brush consists of a number of gaussian brushes normally
    distributed over the canvas.
    """
    diameter = int(round(diameter))
    n = 20

    # Select the positions for the subbrushes.
    positions = np.random.randn(n, 2)
    positions = np.clip(positions, norm.ppf(0.001), norm.ppf(0.999))
    positions = positions / (2.5 * norm.ppf(0.999)) + 0.5
    positions = (positions * diameter).astype(int)

    sizes = np.random.randint(diameter // 3, 2 * diameter // 3, n)

    canvas = np.zeros((diameter, diameter))

    for position, size in zip(positions, sizes):
        dot = brush_gaussian(size)
        draw_alpha(canvas, dot, position)

    return canvas


def brush_calligraphy(diameter, angle=0.25):
    """
    Sample from a calligraphy brush.
    """
    diameter = int(round(diameter))

    image = np.zeros((64, 64))
    image[24:40, 4:60] = 1

    theta = (0.05 * np.random.randn(1)[0] + angle) * np.pi
    A = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    # Apply the transformation multiple times to get a smoothing effect.
    image = transform(image, A, resolution=diameter)
    image = transform(image, np.linalg.inv(A), resolution=diameter)
    image = transform(image, A, resolution=diameter)

    return image


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


def generate_dataset(csv_file, directory, brush, n):
    label = "{};{};{};{};{};{};{};{};{};{};{};{}\n"
    header = label.format('image', 'start_x', 'start_y', 'ctrl_x',
                          'ctrl_y', 'end_x', 'end_y', 'size_start',
                          'size_end', 'red', 'green', 'blue')

    csv_file.write(header)

    for idx in tqdm(range(n)):
        image_name = f'{idx:07d}.png'
        image_path = os.path.join(directory, 'images', image_name)

        canvas = np.zeros((STENCIL_SIZE, STENCIL_SIZE, 3))
        action = np.random.rand(8)
        color = np.random.rand(3)
        brush.draw_stroke(canvas, action, color)

        canvas = (255 * canvas).astype(np.uint8)
        image = Image.fromarray(canvas)
        image.save(image_path)

        action = (255 * action).astype(np.uint8)
        color = (255 * color).astype(np.uint8)
        csv_file.write(label.format(image_name, *action, *color))

    csv_file.close()


def test_strokes():
    print("Create brush")
    brush = Brush(brush_calligraphy)

    plt.figure()
    brush.show_brush_dict()
    plt.close()

    canvas = np.zeros((4 * STENCIL_SIZE, 4 * STENCIL_SIZE, 3))

    # Draw a simple grid.
    canvas[:, ::STENCIL_SIZE // 2, :] = 0.25
    canvas[::STENCIL_SIZE // 2, :, :] = 0.25
    canvas[:, ::STENCIL_SIZE, :] = 1
    canvas[::STENCIL_SIZE, :, :] = 1

    print("Draw strokes")
    for x in range(0, 256, 64):
        for y in range(0, 256, 64):
            action = np.random.rand(8)
            color = np.random.rand(3)

            brush.draw_stroke(canvas, action, color, (x, y))

    plt.figure()
    plt.imshow(canvas)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n', dest='number', type=int, default=1000,
        help="The number of stroke images to create.")
    parser.add_argument(
        '-d', dest='directory', type=str, default='strokes',
        help="")
    parser.add_argument(
        '-t', dest='test', action='store_true',
        help="Display a number of strokes on a test image without saving.")
    args = parser.parse_args()

    if args.test:
        test_strokes()
    else:
        brush = Brush(brush_calligraphy)

        os.makedirs(os.path.join(args.directory, 'images'), exist_ok=True)
        csv_path = os.path.join(args.directory, 'labels.csv')
        csv_file = open(csv_path, 'w')

        try:
            generate_dataset(csv_file, args.directory, brush, args.number)
        except Exception as e:
            csv_file.close()
            raise e
