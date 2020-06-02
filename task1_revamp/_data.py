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

        self.images = [Image.open(f.path) for f in scandir(img_dir)]
        self.targets = None  # TODO
        self.transform = transform

    def __getitem__(self, idx):
        return self.transform(self.images[idx])

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    dataset = TextboxDataset("data/img", "data/box")
    print(dataset[10])
