from os import path, scandir

import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

DEFAULT_RESOLUTION = [448, 224]
DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
    ]
)


class Task1Dataset(Dataset):
    def __init__(
        self,
        img_dir,
        box_dir,
        n_anchor,
        resolution=DEFAULT_RESOLUTION,
        transform=DEFAULT_TRANSFORM,
    ):
        super().__init__()

        self.img_files = list(sorted(scandir(img_dir), key=lambda f: f.name))
        self.box_files = list(sorted(scandir(box_dir), key=lambda f: f.name))

        self.n_anchor = n_anchor
        # heights of anchors have been reduced from the original paper
        # now they are like: [5, 7, 10, 14, 20, ...] (multiply by sqrt(2) each)
        self.anchors = torch.tensor([5 * (2 ** (i / 2)) for i in range(n_anchor)])

        self.resolution = resolution
        self.grid_resolution = [i // 16 for i in resolution]

        self.transform = transform

    def __getitem__(self, idx):
        # process image tensor
        img = Image.open(self.img_files[idx].path).convert("RGB")

        # remember these scaling ratios to scale truth boxes later
        h_scaling = self.resolution[0] / img.height
        w_scaling = self.resolution[1] / img.width

        img = transforms.functional.resize(img, self.resolution)
        img = self.transform(img)

        # target 1: text/non-text classes
        # the elements are {0: non-text, 1: text}
        tgt_1 = torch.zeros(*self.grid_resolution, self.n_anchor, dtype=torch.long)

        # target 2: vertical coordinates
        # the last dimension is [v_c, v_h] in eq.2 of the original paper
        # index 2 marks the locations where target 2 has been filled
        tgt_2 = torch.zeros(*self.grid_resolution, self.n_anchor, 2)
        idx_2 = torch.zeros_like(tgt_2, dtype=torch.bool)

        # target 3: side-refinement offsets
        # the elements are o in eq.4 of the original paper
        # index 3 marks the locations where target 3 has been filled
        tgt_3 = torch.zeros(*self.grid_resolution, self.n_anchor)
        idx_3 = torch.zeros_like(tgt_3, dtype=torch.bool)

        # process target tensors
        with open(self.box_files[idx], "r") as fo:
            for line in fo:
                coordinates = line.strip().split(",", maxsplit=8)
                box = [
                    float(coordinates[0]) * w_scaling,
                    float(coordinates[1]) * h_scaling,
                    float(coordinates[4]) * w_scaling,
                    float(coordinates[5]) * h_scaling,
                ]

                cy_box = (box[1] + box[3]) / 2  # center y of box
                h_box = box[3] - box[1]  # height of box
                # print(h_box)

                # row number: which row of the grid cells is the truth box at
                row_no = int(cy_box // 16)
                # column number: which columns of grid cells does the truth box start and end
                col_no = int(box[0] // 16), int(box[2] // 16)
                # anchor number: which anchor has the closest height to the truth box
                anc_no = (self.anchors - h_box).abs().argmin().item()

                # print(row_no)
                # print(col_no)
                # print(anc_no)

                # set text/non-text classes
                tgt_1[row_no, col_no[0] : col_no[1], anc_no] = 1

                # set vertical coordinates
                cy_anc = row_no * 16 + 8  # center y of anchor
                h_anc = self.anchors[anc_no]  # height of anchor

                v_c = (cy_box - cy_anc) / h_anc
                v_h = torch.log(h_box / h_anc)
                # print(v_c, v_h)

                tgt_2[row_no, col_no[0] : col_no[1], anc_no, 0] = v_c
                tgt_2[row_no, col_no[0] : col_no[1], anc_no, 1] = v_h

                idx_2[row_no, col_no[0] : col_no[1], anc_no, :] = True

                # set side-refinement offsets
                for x_side in [box[0], box[2]]:
                    col_range = torch.arange(
                        round(max((x_side - 32) / 16, 0)),
                        round(min((x_side + 32) / 16, self.grid_resolution[1])),
                    )
                    cx_anc = col_range * 16 + 8
                    o = (x_side - cx_anc) / 16

                    # print(col_range)
                    # print(o)

                    tgt_3[row_no, col_range, anc_no] = o
                    idx_3[row_no, col_range, anc_no] = True

        return img, tgt_1, tgt_2, idx_2, tgt_3, idx_3

    def __len__(self):
        return len(self.img_files)


if __name__ == "__main__":
    import random

    dataset = Task1Dataset("data/img", "data/box", 5, DEFAULT_RESOLUTION)

    data = random.choice(dataset)

    for x in data:
        print(x.shape)
