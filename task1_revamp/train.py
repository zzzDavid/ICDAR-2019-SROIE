import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split

from _data import Task1Dataset
from _model import CtpnModel


def train(model, dataset, device="cpu", batch_size=1, n_epoch=10):
    train_size = 560
    valid_size = len(dataset) - train_size  # 626 - 560 = 66

    print(f"train_size = {train_size}")
    print(f"valid_size = {valid_size}")

    model.to(device)

    criterion_1 = torch.nn.CrossEntropyLoss()
    criterion_2 = torch.nn.SmoothL1Loss(reduction="sum")
    criterion_3 = torch.nn.SmoothL1Loss(reduction="sum")

    # TODO: placeholder hyperparameters
    optimizer = optim.Adagrad(model.parameters(), lr=0.1)
    scheduler = lr_scheduler.ExponentialLR(optimizer, 0.1)

    for i in range(1, n_epoch + 1):
        subsets = random_split(dataset, [train_size, valid_size])

        # training phase
        model.train()
        dataloader = DataLoader(subsets[0], batch_size, shuffle=True, num_workers=4)

        for ii, sample in enumerate(dataloader, start=1):
            img, tgt_1, tgt_2, idx_2, tgt_3, idx_3 = [x.to(device) for x in sample]

            optimizer.zero_grad()

            # outputs of model
            out_1, out_2, out_3 = model.forward(img)

            # print(out_1.shape)
            # print(out_2.shape)
            # print(out_3.shape)

            # losses from eq.5 of the original paper
            loss_1 = criterion_1(out_1.view(-1, 2), tgt_1.view(-1))
            loss_2 = criterion_2(out_2[idx_2], tgt_2[idx_2]).true_divide(
                idx_2.sum().true_divide(2)
            )
            loss_3 = 2 * criterion_3(out_3[idx_3], tgt_3[idx_3]).true_divide(
                idx_3.sum()
            )

            loss = loss_1 + loss_2 + loss_3

            print(f"Epoch {i}.{ii}\n\tLoss: {loss} ({loss_1}, {loss_2}, {loss_3})")

            loss.backward()
            optimizer.step()

        # validation phase
        model.eval()
        dataloader = DataLoader(subsets[1], batch_size=1)

        val_loss = 0

        for ii, sample in enumerate(dataloader, start=1):
            img, tgt_1, tgt_2, idx_2, tgt_3, idx_3 = [x.to(device) for x in sample]
            # outputs of model
            out_1, out_2, out_3 = model.forward(img)

            loss_1 = criterion_1(out_1.view(-1, 2), tgt_1.view(-1))
            loss_2 = criterion_2(out_2[idx_2], tgt_2[idx_2]).true_divide(
                idx_2.sum().true_divide(2)
            )
            loss_3 = 2 * criterion_3(out_3[idx_3], tgt_3[idx_3]).true_divide(
                idx_3.sum()
            )

            loss = loss_1 + loss_2 + loss_3

            val_loss += loss

            # TODO: calculate accuracy

        val_loss = val_loss.true_divide(valid_size)

        print(f"Epoch {i} validation loss: {val_loss}")

        scheduler.step()


if __name__ == "__main__":
    n_anchor = 5
    resolution = [448, 224]

    model = CtpnModel(n_anchor)
    dataset = Task1Dataset("data/img", "data/box", n_anchor, resolution)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    if device == "cpu":
        train(model, dataset, device, batch_size=1)
    else:
        train(model, dataset, device, batch_size=8)
