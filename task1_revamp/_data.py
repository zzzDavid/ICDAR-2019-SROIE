from os import path, scandir

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

INPUT_RESOLUTION = torch.tensor([448, 224])
GRID_RESOLUTION = INPUT_RESOLUTION // 16

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

        self.anchors = torch.tensor([11 // (0.7 ** i) for i in range(n_anchor)])

    def __getitem__(self, idx):
        # process image tensor
        img = Image.open(self.img_files[idx].path)

        # remember these scaling ratios to scale truth boxes later
        h_scaling = INPUT_RESOLUTION[0] / img.height
        w_scaling = INPUT_RESOLUTION[1] / img.width

        img = self.transform(img)

        # target 1: text/non-text classes
        # the elements are {0: non-text, 1: text}
        tgt_1 = torch.zeros(*GRID_RESOLUTION, self.n_anchor)

        # target 2: vertical coordinates
        # the last dimension is [v_c, v_h] in eq.2 of the original paper
        tgt_2 = torch.zeros(*GRID_RESOLUTION, self.n_anchor, 2)

        # target 3: side-refinement offsets
        # the elements are o in eq.4 of the original paper
        tgt_3 = torch.zeros(*GRID_RESOLUTION, self.n_anchor)

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

                # row number: which row of the grid cells is the truth box at
                row_no = cy_box // 16
                # column number: which columns of grid cells does the truth box start and end
                col_no = torch.tensor([box[0] // 16, box[2] // 16])
                # anchor number: which anchor has the closest height to the truth box
                anc_no = (self.anchors - h_box).abs().argmin()

                # set text/non-text classes
                tgt_1[row_no, col_no[0] : col_no[1], anc_no] = 1

                # set vertical coordinates
                cy_anc = row_no * 16 + 8  # center y of anchor
                h_anc = self.anchors[anc_no]  # height of anchor

                v_c = (cy_box - cy_anc) / h_anc
                v_h = torch.log(h_box / h_anc)

                tgt_2[row_no, col_no[0] : col_no[1], anc_no, 0] = v_c
                tgt_2[row_no, col_no[0] : col_no[1], anc_no, 1] = v_h

                # set side-refinement offsets
                x_side = torch.tensor([box[0], box[2]])  # horizontal sides of truth box
                cx_anc = col_no * 16 + 8  # center x of anchors

                o = (x_side - cx_anc) / 16

                tgt_3[row_no, col_no, :] = o

        return img, tgt_1, tgt_2, tgt_3

    def __len__(self):
        return len(self.img_files)


if __name__ == "__main__":
    x = torch.tensor([448, 224])
    y = x // 16

    c = torch.zeros(*y, 2)

    print(c.size())
