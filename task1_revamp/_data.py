from os import path, scandir

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        transforms.RandomPerspective(fill=1),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
    ]
)


class TextboxDataset(Dataset):
    def __init__(self, img_dir, box_dir, transform=DEFAULT_TRANSFORM):
        super().__init__()

        self.images = list(sorted(scandir(img_dir), key=lambda f: f.name))
        self.targets = list(sorted(scandir(box_dir), key=lambda f: f.name))

        self.transform = transform

    def __getitem__(self, idx):
        # process image tensor
        img = Image.open(self.images[idx].path)
        img_tensor = self.transform(img)

        # process target tensor
        with open(self.targets[idx], "r") as fo:
            for line in fo:
                coordinates = line.strip().split(",", maxsplit=8)[:8]
                coordinates = [float(c) for c in coordinates]
                # TODO: put coordinates in a tensor

        return img_tensor, coordinates

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    dataset = TextboxDataset("data/img", "data/box")
    print(dataset.images)

    img_tensor, coordinates = dataset[1]
    print(coordinates)
