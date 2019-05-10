import numpy as np
import torch


class ToTensor(object):
    def __call__(self, sample):
        image, params = sample['image'], sample['parameters']

        image = image.transpose((2, 0, 1)).astype(float) / 255
        params /= 255

        return {
            'image': torch.from_numpy(image),
            'parameters': torch.from_numpy(params)
        }
