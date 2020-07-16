import torch
import torchvision


class CtpnModel(torch.nn.Module):
    def __init__(self, n_anchor=10):
        super().__init__()

        self.n_anchor = n_anchor

        # feature extractor from VGG16
        self.features = torchvision.models.vgg16_bn(pretrained=True).features[:-1]
        # the sliding window upon the last features of VGG16
        self.slider = torch.nn.Conv2d(512, 512, 3, padding=1)
        # bi-directional LSTM
        self.bilstm = torch.nn.LSTM(512, 128, bidirectional=True)
        # fully connected output 1: text/non-text scores
        self.fc_1 = torch.nn.Linear(256, n_anchor * 2)
        # fully connected output 2: vertical coordinates
        self.fc_2 = torch.nn.Linear(256, n_anchor * 2)
        # fully connected output 3: side-refinement offsets
        self.fc_3 = torch.nn.Linear(256, n_anchor)

    def forward(self, x):
        """CTPN model forward function

        Inputs:
            x -- of shape N x C x H x W, typically N x 3 x 448 x 224. Image tensor.

        Outputs:
            y_1 -- of shape N x H x W x k x 2. Predicted text/non-text scores.
            y_2 -- of shape N x H x W x k x 2. Predicted vertical coordinates.
            y_3 -- of shape N x H x W x k. Predicted side-refinement offsets.
        """
        # x: N x 3 x 448 x 224
        x = self.features(x)
        # x: N x 512 x 28 x 14
        x = self.slider(x)
        # x: N x 512 x 28 x 14

        x = x.permute(2, 3, 0, 1)
        # x: 28 x 14 x N x 512

        # use BiLSTM on each row and collect them in c
        c = []
        for a in x:
            b, _ = self.bilstm(a)
            c.append(b)

        c = torch.stack(c)
        # c: 28 x 14 x N x 256

        c = c.permute(2, 0, 1, 3)
        # c: N x 28 x 14 x 256

        y_1 = self.fc_1(c)
        y_1 = y_1.reshape(*y_1.shape[:3], self.n_anchor, 2)
        # y_1: N x 28 x 14 x k x 2, text/non-text scores
        y_2 = self.fc_2(c)
        y_2 = y_2.reshape(*y_2.shape[:3], self.n_anchor, 2)
        # y_2: N x 28 x 14 x k x 2, vertical coordinates
        y_3 = self.fc_3(c)
        # y_3: N x 28 x 14 x k, side-refinement offsets

        return y_1, y_2, y_3


if __name__ == "__main__":
    # test code
    model = CtpnModel()

    x = torch.randn(1, 3, 448, 224)
    y_1, y_2, y_3 = model.forward(x)

    print(y_1.shape)
    print(y_2.shape)
    print(y_3.shape)

    # bilstm = torch.nn.LSTM(512, 128, bidirectional=True)

    # x = torch.randn(14, 1, 512)
    # y, h = bilstm(x)

    # print(repr(h))
