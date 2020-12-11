import torch
import torch.nn.functional as F
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=2):
        super(ResNet, self).__init__()
        self.inchannel = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 16,  1, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 32,  1, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 64, 1, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 128, 1, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(13824, 128)
        self.fc2 = nn.Linear(128, num_classes)
        # self.fc1 = nn.Linear(27648, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # print(out.shape)
        out = F.avg_pool2d(out, 4)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        out = self.fc2(self.dropout(self.relu(self.fc1(out))))
        # out = self.fc1(out)
        return out


# class Classifier(torch.nn.Module):
#     class Block(torch.nn.Module):
#         def __init__(self, n_input, n_output, kernel_size=3, stride=2):
#             super().__init__()
#             self.c1 = torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
#                                       stride=stride, bias=False)
#             self.c2 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
#             self.c3 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
#             self.b1 = torch.nn.BatchNorm2d(n_output)
#             self.b2 = torch.nn.BatchNorm2d(n_output)
#             self.b3 = torch.nn.BatchNorm2d(n_output)
#             self.skip = torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)

#         def forward(self, x):
#             return F.relu(self.b3(self.c3(F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x)))))))) + self.skip(x))

#     def __init__(self, layers=[16, 32, 64, 128], n_output_channels=2, kernel_size=3):
#         super().__init__()
#         self.input_mean = torch.Tensor([0.3521554, 0.30068502, 0.28527516])
#         self.input_std = torch.Tensor([0.18182722, 0.18656468, 0.15938024])

#         L = []
#         c = 3
#         for l in layers:
#             L.append(self.Block(c, l, kernel_size, 2))
#             c = l
#         self.network = torch.nn.Sequential(*L)
#         self.classifier = torch.nn.Linear(c, n_output_channels)

#     def forward(self, x):
#         z = self.network((x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device))
#         return self.classifier(z.mean(dim=[2, 3]))


def save_model(model, name=None):
    from torch import save
    from os import path
    # Save model based on the naming
    if name is None:
        if isinstance(model, ResNet):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'classifier_res.th'))
    else:
        if isinstance(model, ResNet):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '{0}_resnet.th'.format(name)))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    # TODO: Modify this function so that it would fit for different models
    # Load the ResNet model
    from torch import load
    from os import path
    r = ResNet(ResidualBlock)
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'classifier_res.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    from .controller import control
    from .utils import PyTux
    from argparse import ArgumentParser

    # Test classifier
    def test_classifier(args):
        # Load model
        classifier = load_model().eval()
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, classifier=classifier, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()

    parser = ArgumentParser("Test the classifier")
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_classifier(args)
