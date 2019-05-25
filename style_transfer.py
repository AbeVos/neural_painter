import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from PIL import Image
from torchvision.utils import save_image

from architectures.vae import VAE

def load_image(path, img_size):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()])

    image = Image.open(path)
    image = transform(image).unsqueeze(0)

    return image


def gram_matrix(features):
    features = features.view(128, -1)
    gram = features @ features.t()
    return gram.unsqueeze(0)


class StyleLoss(nn.Module):
    def __init__(self, style_model, style_img):
        super(StyleLoss, self).__init__()

        self.model = style_model
        self.mse = nn.MSELoss()
        self.style = style_model(style_img)
        self.normalize = self.style.nelement()
        self.style = gram_matrix(self.style).detach()

    def forward(self, x):
        style = self.model(x)
        style = gram_matrix(style)

        loss = self.mse(style, self.style) / self.normalize

        return loss


class ContentLoss(nn.Module):
    def __init__(self, content_model, content_img):
        super(ContentLoss, self).__init__()

        self.model = content_model
        self.mse = nn.MSELoss()
        self.content = content_model(content_img).detach()

    def forward(self, x):
        content = self.model(x)
        loss = self.mse(content, self.content)

        return loss


if __name__ == "__main__":
    img_size = 512
    device = torch.device('cuda:0')

    vae = VAE(latent_dim=512, device=device)
    vae.load_state_dict(torch.load("models/content_net.pth"))
    vae = vae.eval()
    model = vae.encoder.layers

    for layer in model:
        if isinstance(layer, nn.Conv2d):
            weights = layer.weight.data

            save_image(weights, "content_weights_layer1.png", nrow=16,
                       normalize=True)
            
            break

    style_weight = [5, 4, 1]
    style_models = [model[:1], model[:3], model[:5]]
    content_model = model

    style_img = load_image('images/vangogh.png', img_size).to(device)
    content_img = load_image('images/einstein.jpg', img_size).to(device)

    style_loss = [StyleLoss(model, style_img) for model in style_models]
    content_loss = ContentLoss(content_model, content_img)
    
    # Mix the content image with some noise to get a begin state.
    alpha = 0.5
    image = alpha * content_img.clone() \
         + (1 - alpha) * torch.rand(1, 3, img_size, img_size).to(device)
    optimizer = optim.LBFGS([image.requires_grad_()])

    for step in range(10):
        def closure():
            optimizer.zero_grad()
            image.data.clamp_(0, 1)

            style = [weight * loss(image) for weight, loss
                     in zip(style_weight, style_loss)]
            style = torch.stack(style).sum()
            loss = style + 1e-5 * content_loss(image)

            loss.backward()

            print(loss.item())

            return loss

        optimizer.step(closure)
        image.data.clamp_(0, 1)

        # print(f"Step {step}: {loss.item()}")
        print(f"Step {step+1}")

        images = torch.cat((content_img, image, style_img), dim=0)
        save_image(images, "samples/style_transfer.png", nrow=3)
