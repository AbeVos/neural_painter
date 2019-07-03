import torch
import torchvision.transforms as transforms

from PIL import Image
from torchvision.utils import save_image


def load_image(path, img_size):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()])

    image = Image.open(path)
    image = transform(image).unsqueeze(0)

    return image


content_img = load_image('images/einstein.jpg', 512)

content_param = torch.rfft(content_img, 2, normalized=True, onesided=False)
# content_param = (content_param ** 2).sum(-1).sqrt()
content_img = torch.irfft(content_param, 2, True, False)

print(content_param.shape, content_img.shape)
save_image(content_img, 'samples/style_param.png')
