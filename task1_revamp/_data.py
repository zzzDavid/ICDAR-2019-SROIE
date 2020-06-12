from os import path, scandir

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

INPUT_RESOLUTION = [448, 224]
GRID_RESOLUTION = [i // 16 for i in INPUT_RESOLUTION]

DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(INPUT_RESOLUTION),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        transforms.RandomPerspective(fill=1),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
    ]
)


class TextboxDataset(Dataset):
    def __init__(self, img_dir, box_dir, n_anchor, transform=DEFAULT_TRANSFORM):
        super().__init__()

        self.img_files = list(sorted(scandir(img_dir), key=lambda f: f.name))
        self.box_files = list(sorted(scandir(box_dir), key=lambda f: f.name))

        self.n_anchor = n_anchor
        self.transform = transform

    def __getitem__(self, idx):
        # process image tensor
        img = Image.open(self.img_files[idx].path)
        h_scaling = INPUT_RESOLUTION[0] / img.height
        w_scaling = INPUT_RESOLUTION[1] / img.width

        img = self.transform(img)

        # target 1: vertical coordinates
        tgt_1 = torch.zeros(*GRID_RESOLUTION, 2 * self.n_anchor)
        # target 2: text/non-text scores
        tgt_2 = torch.zeros(*GRID_RESOLUTION, 2 * self.n_anchor)
        # target 3: side-refinement offsets
        tgt_3 = torch.zeros(*GRID_RESOLUTION, self.n_anchor)

        # process target tensors
        with open(self.box_files[idx], "r") as fo:
            for line in fo:
                coordinates = line.strip().split(",", maxsplit=8)[:8]
                truth_box = [
                    float(coordinates[0]) * w_scaling,
                    float(coordinates[1]) * h_scaling,
                    float(coordinates[4]) * w_scaling,
                    float(coordinates[5]) * h_scaling,
                ]
                # TODO

        return img, tgt_1, tgt_2, tgt_3

    def __len__(self):
        return len(self.img_files)


if __name__ == "__main__":
    # dataset = TextboxDataset("data/img", "data/box")
    # print(dataset.images)

    # img, coordinates = dataset[1]
    # print(coordinates)

    print(GRID_RESOLUTION)
