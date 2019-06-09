import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm

from curves import Bezier

STENCIL_SIZE = 64


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


def draw_stroke(canvas, action, color, offset=(0, 0)):
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
    action = (STENCIL_SIZE * np.array(action)).astype(int)
    positions = action[:6].reshape(-1, 2) + np.array(offset)[None, :]
    start, ctrl, end = positions
    start_size, end_size = np.clip(action[6:], 1, STENCIL_SIZE)

    curve = Bezier(start, ctrl, end)
    alphamap = np.zeros_like(canvas[..., 0])

    # Draw alphamap.
    for t in np.linspace(0, 1, int(curve.length)):
        position = curve.evaluate(t).astype(int)
        size = (1 - t) * start_size + t * end_size

        alpha = brush_gaussian(int(size))
        draw_alpha(alphamap, alpha, position)

    # Draw color to the canvas based on the generated alphamap.
    canvas = np.copyto(canvas,
        alphamap[..., None] * color[None, None, :] \
        + (1 - alphamap[..., None]) * canvas)


if __name__ == "__main__":
    canvas = np.zeros((128, 256, 3))

    # Draw a simple grid.
    canvas[:, ::STENCIL_SIZE // 2, :] = 0.25
    canvas[::STENCIL_SIZE // 2, :, :] = 0.25
    canvas[:, ::STENCIL_SIZE, :] = 1
    canvas[::STENCIL_SIZE, :, :] = 1

    for offset in [(0, 0), (0, 64), (64, 0), (64, 64)]:
        action = np.random.rand(8)
        action[-2:] = np.clip(action[-2:], 0.1, 1)

        draw_stroke(canvas, action, np.random.rand(3), offset)

    plt.imshow(canvas)
    plt.show()
