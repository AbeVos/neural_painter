import random
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm

from curves import Bezier

STENCIL_SIZE = 64


class Brush():
    def __init__(self, brush, stencil_size=STENCIL_SIZE, n_per_size=10):
        self.min_size = stencil_size // 8
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

        for idx, size in enumerate(range(self.min_size, self.max_size), 2):
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
    diameter = int(round(diameter))
    n = 20

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


if __name__ == "__main__":
    print("Create brush")
    brush = Brush(brush_paint)

    canvas = np.zeros((4 * STENCIL_SIZE, 4 * STENCIL_SIZE, 3))

    # Draw a simple grid.
    canvas[:, ::STENCIL_SIZE // 2, :] = 0.25
    canvas[::STENCIL_SIZE // 2, :, :] = 0.25
    canvas[:, ::STENCIL_SIZE, :] = 1
    canvas[::STENCIL_SIZE, :, :] = 1

    print("Draw strokes")
    for offset in [
            (STENCIL_SIZE, STENCIL_SIZE),
            (STENCIL_SIZE, 2 * STENCIL_SIZE),
            (2 * STENCIL_SIZE, STENCIL_SIZE),
            (2 * STENCIL_SIZE, 2 * STENCIL_SIZE)]:
        action = np.random.rand(8)
        color = np.random.rand(3)

        brush.draw_stroke(canvas, action, color, offset)

    plt.imshow(canvas)
    plt.axis('off')
    plt.show()
