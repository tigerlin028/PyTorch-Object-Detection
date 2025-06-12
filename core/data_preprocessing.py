from ..transforms.transforms import *


def normalize_transform(img, boxes=None, labels=None, std=1.0):
    return img / std, boxes, labels

class Normalize:
    def __init__(self, std=1.0):
        self.std = std

    def __call__(self, img, boxes=None, labels=None):
        return normalize_transform(img, boxes, labels, std=self.std)


class TrainAugmentation:
    def __init__(self, size, mean=0, std=1.0):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean),
            Normalize(std),
            ToTensor(),
        ])

    def __call__(self, img, boxes, labels):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        return self.augment(img, boxes, labels)


class TestTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            ToPercentCoords(),
            Resize(size),
            SubtractMeans(mean),
            Normalize(std),
            ToTensor(),
        ])

    def __call__(self, image, boxes, labels):
        return self.transform(image, boxes, labels)


class PredictionTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            Resize(size),
            SubtractMeans(mean),
            Normalize(std),
            ToTensor()
        ])

    def __call__(self, image):
        image, _, _ = self.transform(image)
        return image