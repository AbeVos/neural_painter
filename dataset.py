import os

from skimage import io
from torch.utils.data import Dataset


class BrushStrokesDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.stroke_params = read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.stroke_params)

    def __getitem__(self, idx):
        image_name = os.path.join(
            self.root_dir, self.stroke_params['image'][idx])
        image = io.imread(image_name)


def read_csv(path):
    pass
