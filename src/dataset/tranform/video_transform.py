import torch
from torchvision.transforms import (
    Resize,
    ColorJitter,
    Compose,
    GaussianBlur,
    RandomAffine,
    RandomApply,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomPerspective,
    RandomSolarize,
)


class VideoTransform:
    def __init__(self):
        self.transform = Compose(
            [
                RandomAffine((0.9, 1.1)),
                RandomColorJitterFace(
                    (0.9, 1.1), (0.9, 1.1), (0.9, 1.1), (-0.1, 0.1), 0.8
                ),
                RandomHorizontalFlip(0.5),
                RandomGrayscale(0.2),
                RandomPerspective(0.5, 0.5),
                RandomGaussianBlur(0.2, (5, 5), (0.01, 2.0)),
                RandomSolarize(0.6, 0.4),
                # Resize(224),
            ]
        )

    def __call__(self, x):
        return self.transform(x)


class RandomColorJitterFace:
    """Apply ColorJitter to the face region only"""

    def __init__(self, brightness, contrast, saturation, hue, p):
        self.color_jitter = RandomApply(
            [ColorJitter(brightness, contrast, saturation, hue)], p=p
        )

    def __call__(self, x):
        mask = torch.where(x > 0, True, False)
        transformed = self.color_jitter(x) * mask
        return transformed


class RandomGaussianBlur:
    def __init__(self, p, kernel_size, sigma):
        self.gaussian = RandomApply([GaussianBlur(kernel_size, sigma)], p=p)

    def __call__(self, x):
        transformed = self.gaussian(x)
        return transformed


class ChalearnVideoTransform:
    def __init__(self):
        self.transform = Compose([Resize((112, 112), antialias=True)])

    def __call__(self, x):
        return self.transform(x)
