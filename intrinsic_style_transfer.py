import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from PIL import Image
from torchvision.utils import save_image

from train_action_encoder import create_images, blend
from architectures.vae import VAE
from architectures.painters import VAEPainter, ActionEncoder
from strokes import draw_stroke


def load_image(path, img_size):
    """Load an image from path to a Tensor and resize it."""
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()])

    image = Image.open(path)
    image = transform(image).unsqueeze(0)

    return image


def sliding_window(image, step_size, window_size):
    """Slide a window across the image."""
    for y in range(0, image.shape[2], step_size):
        if y + window_size > image.shape[2]:
            continue

        for x in range(0, image.shape[3], step_size):
            if x + window_size > image.shape[3]:
                continue

            # Yield the current window.
            yield x, y, image[..., y:y + window_size, x:x + window_size]


def random_window(image, window_size, n, distribution='normal'):
    """Generate random window positions."""
    image_size = np.array(image.shape[2:])

    for i in range(n):
        if distribution == 'normal':
            center = np.random.randn(2) * 70
            center += image_size // 2
            top_left = np.round(center - window_size / 2).astype(int)
            btm_right = np.round(center + window_size / 2).astype(int)

            if top_left[0] < 0:
                btm_right[0] -= top_left[0]
                top_left[0] -= top_left[0]
            elif btm_right[0] >= image_size[0]:
                top_left[0] -= btm_right[0] - image_size[0]
                btm_right[0] -= btm_right[0] - image_size[0]

            if top_left[1] < 0:
                btm_right[1] -= top_left[1]
                top_left[1] -= top_left[1]
            elif btm_right[1] >= image_size[1]:
                top_left[1] -= btm_right[1] - image_size[1]
                btm_right[1] -= btm_right[1] - image_size[1]

        y, x = top_left
        y_, x_ = btm_right
        segment = image[..., y:y_, x:x_]

        yield x, y, segment


class ContentLoss(nn.Module):
    def __init__(self, content_model, content_img):
        super(ContentLoss, self).__init__()

        self.model = content_model
        self.mse = nn.MSELoss()
        self.content, logvar = content_model(content_img)
        self.content = self.content.detach()
        logvar = logvar.detach()

    def forward(self, x):
        content, logvar = self.model(x)
        logvar = logvar.detach()
        loss = self.mse(content, self.content)

        return loss


def pastel_loss(colors):
    maximum, _ = torch.max(colors, dim=1)
    minimum, _ = torch.min(colors, dim=1)
    saturation = (maximum - minimum) / (maximum + 1e-8)
    return torch.mean(torch.abs(0.5 - saturation) ** 2
                      + torch.abs(1 - maximum) ** 2)


def paint_segment(content, dst=None, n=10, steps=500, device='cuda:0'):
    """
    Paint a picture segment.
    """
    content_loss = ContentLoss(content_model, content)

    actions = torch.rand(n, 8).to(device)
    actions[:, -2:] *= 0.5  # Limit stroke sizes.
    colors = torch.rand(n, 3).to(device)

    canvas = torch.empty((1, 3, 64, 64)).to(device)

    if dst is None:
        dst = torch.ones_like(canvas)

    optimizer = optim.Adam(
        [actions.requires_grad_(), colors.requires_grad_()], lr=3e-2)
    # optimizer = optim.Adam(
    #     [actions.requires_grad_()], lr=3e-2)

    for step in range(steps):
        actions.data.clamp_(0, 1)
        actions.data[-2:].clamp_(0.1, 0.3)
        colors.data.clamp_(0, 1)

        canvas.copy_(dst).detach_()

        # Generate strokes from the actions.
        strokes = action_encoder(actions)
        strokes = torch.sigmoid(decoder(strokes))
        strokes = create_images(strokes, colors, device)

        # Blend the stroke unto the canvas.
        for stroke in strokes:
            canvas = blend(stroke[None, ...], canvas)

        loss = content_loss(canvas) \
                + 0.5 * nn.functional.mse_loss(canvas, content)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if step % 100 == 0:
            print(step, loss.item())

            images = torch.cat((content, canvas), dim=0)
            save_image(images, "samples/intrinsic_style.png", nrow=2)

    actions.data.clamp_(0, 1)
    actions.data[-2:].clamp_(0.1, 0.3)
    colors.data.clamp_(0, 1)

    return canvas.detach().cpu(), \
            (actions.detach().cpu(), colors.detach().cpu())


if __name__ == "__main__":
    img_size = 512
    device = torch.device('cuda:0')

    # Prepare models.
    vae = VAE(latent_dim=512, device=device).eval()
    vae.load_state_dict(torch.load("models/content_net.pth"))
    vae = vae.eval()
    content_model = vae.encoder

    action_encoder = ActionEncoder(8, 8, 256, 5).to(device).eval()
    action_encoder.load_state_dict(torch.load("models/action_encoder.pth"))

    painter = VAEPainter(8, device).to(device).eval()
    painter.load_state_dict(torch.load("models/painter.pth"))
    decoder = painter.decoder

    content_img = load_image('images/vuurtoren.jpg', img_size).to(device)

    # canvas = torch.ones_like(content_img)
    height, width = content_img.shape[2:]
    canvas = np.ones((height, width, 3))

    for x, y, content_segment in sliding_window(content_img, 32, 64):
    # for x, y, segment in random_window(content_img, 64, 1000):
        # segment = canvas[..., y:y+64, x:x+64]
        target_segment = torch.Tensor(canvas[y:y+64, x:x+64]).to(device)
        target_segment = target_segment.permute(2, 0, 1).unsqueeze(0)

        painted_segment, actions = paint_segment(
            content_segment, dst=target_segment, n=15, steps=500)

        # canvas[..., y:y+64, x:x+64] = painted_segment

        for action, color in zip(*actions):
            draw_stroke(canvas, action.numpy(), color.numpy(), offset=(x, y))

        # save_image(canvas, "samples/painting.png", nrow=1)
        plt.figure()
        plt.imshow(canvas)
        plt.axis('off')
        plt.savefig("samples/painting.png")
        plt.close()
