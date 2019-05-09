import os
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from PIL import Image

images = []

for path in tqdm(os.listdir('images')):
    image = Image.open(os.path.join('images', path))
    image.load()
    image = np.asarray(image, dtype="uint8")
    images.append(image)

image = np.mean(images, 0, int)

plt.imshow(image)
plt.show()
