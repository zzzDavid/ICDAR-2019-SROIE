import torch
import torchvision


class CtpnModel(torch.nn.Module):
    def __init__(self, n_anchor):
        super().__init__()

        # feature extractor from VGG16
        self.features = torchvision.models.vgg16_bn(pretrained=True).features[:-1]
        # the sliding window upon the last features of VGG16
        self.slider = torch.nn.Conv2d(512, 512, 3, padding=1)
        # bi-directional LSTM
        self.blstm = torch.nn.LSTM(256, 256, bidirectional=True)
        # fully connected output 1: vertical coordinates
        self.fc_1 = torch.nn.Linear(256, 2 * n_anchor)
        # fully connected output 2: confidence scores
        self.fc_2 = torch.nn.Linear(256, 2 * n_anchor)
        # fully connected output 3: side-refinement offsets
        self.fc_3 = torch.nn.Linear(256, n_anchor)

    def forward(self, x):
        x = self.features(x)
        x = self.slider(x)
        # TODO


if __name__ == "__main__":
    model = CtpnModel()
    print(model)

    x = torch.randn(1, 3, 256, 512)
    print(x.shape)
    print(model.forward(x).shape)
