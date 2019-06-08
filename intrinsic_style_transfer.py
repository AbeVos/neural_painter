import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from PIL import Image
from torchvision.utils import save_image

from train_action_encoder import create_images, blend
from architectures.vae import VAE
from architectures.painters import VAEPainter, ActionEncoder

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
    for y in range(0, image.shape[2] - window_size, step_size):
        if y + window_size > image.shape[2]:
            continue

        for x in range(0, image.shape[3] - window_size, step_size):
            if x + window_size > image.shape[3]:
                continue

            # Yield the current window.
            yield x, y, image[..., y:y + window_size, x:x + window_size]


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
        dst = torch.ones((1, 3, 64, 64))

    optimizer = optim.Adam(
        [actions.requires_grad_(), colors.requires_grad_()], lr=3e-2)

    for step in range(steps):
        actions.data.clamp_(0, 1)

        canvas.copy_(dst).detach_()

        # Generate strokes from the actions.
        strokes = action_encoder(actions)
        strokes = torch.sigmoid(decoder(strokes))
        strokes = create_images(strokes, colors, device)

        # Blend the stroke unto the canvas.
        for stroke in strokes:
            canvas = blend(stroke[None, ...], canvas)

        loss = content_loss(canvas) \
                + nn.functional.mse_loss(canvas, content)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if step % 100 == 0:
            print(step, loss.item())

            images = torch.cat((content, canvas), dim=0)
            save_image(images, "samples/intrinsic_style.png", nrow=2)

    return canvas.detach().cpu(), actions.detach().cpu()


if __name__ == "__main__":
    img_size = 64
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

    painting = torch.ones_like(content_img)

    for x, y, segment in sliding_window(content_img, 16, 64):
        canvas = painting[..., y:y+64, x:x+64]
        print(x, x+64)
        painted_segment, actions = paint_segment(segment, dst=canvas, n=10,
                                                 steps=100)

        painting[..., y:y+64, x:x+64] = painted_segment
        save_image(painting, "samples/painting.png", nrow=1)
