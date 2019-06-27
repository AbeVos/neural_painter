import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from scipy import misc
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

try:
    from .transform import ToTensor
except ModuleNotFoundError:
    from transform import ToTensor


class BrushStrokeDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.stroke_params = read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.stroke_params['image'])

    def __getitem__(self, idx):
        # Load image.
        image_name = os.path.join(
            self.root_dir, self.stroke_params['image'][idx])
        image = misc.imread(image_name)

        # Load stroke parameters.
        param_keys = [key for key in list(self.stroke_params.keys())[1:]]
        action = np.array(
            [self.stroke_params[key][idx] for key in param_keys], np.uint8)
        action = action.astype(float).T

        sample = {'image': image, 'action': action}

        # Transform sample.
        if self.transform:
            sample = self.transform(sample)

        return sample


def read_csv(path):
    data = defaultdict(list)

    with open(path) as file:
        reader = csv.DictReader(file, delimiter=';')

        for row in reader:
            for key, value in row.items():
                try:
                    data[key].append(int(value))
                except ValueError:
                    data[key].append(value)

    return data


def show_data(n_images, nrow=5):
    dataloader = DataLoader(dataset, batch_size=n_images, shuffle=True,
                            num_workers=4)

    plt.figure()

    batch = next(iter(dataloader))
    images = batch['image']

    for idx, image in enumerate(images):
        image = image.permute(1, 2, 0)

        plt.subplot(n_images // nrow, nrow, idx+1)
        plt.imshow(image)
        plt.axis('off')

    plt.show()


if __name__ == "__main__":
    dataset = BrushStrokeDataset(
        'calligraphy_100k/labels.csv', 'calligraphy_100k/images/',
        transform=transforms.Compose([
            ToTensor()
        ]))

    show_data(25)
