import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np
import torch

__all__ = ['TransformPair']


class TransformPair(object):
    """
    Class to ensure image and locations get the same transformations during training
    """

    def __init__(self, output_size, train):
        self.output_size = output_size
        self.train = train

    def __call__(self, image, ground_truth):
        if self.train:
            # shape augmentations
            # left-right mirroring
            if np.random.random() > 0.5:
                image = TF.hflip(image)
                ground_truth = TF.hflip(ground_truth)

            # up-down mirroring
            if np.random.random() > 0.5:
                image = TF.vflip(image)
                ground_truth = TF.vflip(ground_truth)

            # random rotation
            angle = np.random.uniform(-180, 180)
            image = TF.rotate(image, angle, expand=True)
            ground_truth = TF.rotate(ground_truth, angle, expand=True)

            # center crop
            center_crop = transforms.CenterCrop(int(self.output_size * 1.5))
            image = center_crop(image)
            ground_truth = center_crop(ground_truth)

            # random resized crop
            if np.random.random() > 0.2:
                # random crop
                i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=(0.2, 0.9), ratio=(1, 1))
                image = TF.resized_crop(image, i, j, h, w, size=int(self.output_size))
                ground_truth = TF.resized_crop(ground_truth, i, j, h, w, size=int(self.output_size))

            # random crop without resize
            else:
                i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.output_size, self.output_size))
                image = TF.crop(image, i, j, h, w)
                ground_truth = TF.crop(ground_truth, i, j, h, w)

            # color augmentations
            for col_aug in [TF.adjust_contrast, TF.adjust_brightness, TF.adjust_saturation, TF.adjust_gamma]:
                if np.random.random() > 0.5:
                    adjust_factor = np.random.uniform(0.5, 1.5)
                    image = col_aug(image, adjust_factor)

            if np.random.random() > 0.5:
                hue_factor = np.random.uniform(-0.15, 0.15)
                image = TF.adjust_hue(image, hue_factor)

        else:
            center_crop = transforms.CenterCrop(self.output_size)
            image = center_crop(image)
            ground_truth = center_crop(ground_truth)

        # change locations to tensor
        ground_truth = (np.array(ground_truth) > 0).astype(np.float32)
        area = (np.sum(ground_truth) / len(ground_truth.flatten()) * 100).astype(np.float32)
        ground_truth = torch.Tensor(ground_truth.reshape([1, self.output_size, self.output_size]))
        image = TF.normalize(TF.to_tensor(image), [0.5] * 3, [0.25] * 3)

        return image, ground_truth, area
