from _data import Task1Dataset
from _model import CtpnModel

from torch.utils.data import DataLoader
import torch


def train(model, dataloader):
    for i_batch, batch_sample in enumerate(dataloader):
        print(i_batch)
        for x in batch_sample:
            print(x.shape)


if __name__ == "__main__":
    n_anchor = 5
    resolution = [448, 224]

    model = CtpnModel(n_anchor)
    dataset = Task1Dataset("data/img", "data/box", n_anchor, resolution)
    dataloader = DataLoader(
        dataset, batch_size=10, num_workers=4, shuffle=True, drop_last=True
    )

    train(model, dataloader)
