import argparse

import torch
from torch import nn, optim

from my_data import VOCAB, MyDataset, color_print
from my_models import MyModel0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", default="cuda")
    parser.add_argument("-b", "--batch_size", type=int, default=8)
    parser.add_argument("-e", "--max_epoch", type=int, default=2000)
    parser.add_argument("-v", "--val-at", type=int, default=100)
    parser.add_argument("-i", "--hidden-size", type=int, default=512)

    args = parser.parse_args()
    args.device = torch.device(args.device)

    # torch.backends.cudnn.enabled = False

    model = MyModel0(len(VOCAB), 20, args.hidden_size).to(args.device)

    dataset = MyDataset("data/data_dict.pth", args.device)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 1, 1.2, 0.8, 5], device=args.device))
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1000)

    for i in range(args.max_epoch // args.val_at):
        train(
            model,
            dataset,
            criterion,
            optimizer,
            (i * args.val_at + 1, (i + 1) * args.val_at + 1),
            args.batch_size,
        )
        validate(model, dataset)

    validate(model, dataset, batch_size=10)


def validate(model, dataset, batch_size=1):
    model.eval()
    with torch.no_grad():
        keys, text, truth = dataset.get_val_data(batch_size=batch_size)

        pred = model(text)

        for i, key in enumerate(keys):
            print_text, _ = dataset.val_dict[key]
            print_text_class = pred[:, i][: len(print_text)].cpu().numpy()
            color_print(print_text, print_text_class)


def train(model, dataset, criterion, optimizer, epoch_range, batch_size):
    model.train()

    for epoch in range(*epoch_range):
        optimizer.zero_grad()

        text, truth = dataset.get_train_data(batch_size=batch_size)
        pred = model(text)

        loss = criterion(pred.view(-1, 5), truth.view(-1))
        loss.backward()

        optimizer.step()

        print("#{:04d} | Loss: {:.4f}".format(epoch, loss.item()) )


if __name__ == "__main__":
    main()
