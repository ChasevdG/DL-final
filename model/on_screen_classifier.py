import torch
import torch.nn.functional as F
import torch.nn as nn


class On_Screen_Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(3, 64, 5)
        self.conv2 = torch.nn.Conv2d(64, 64, 5)

        self.dropout = nn.Dropout()
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(446976, 128)
        self.fc2 = torch.nn.Linear(128, 32)
        self.fc3 = torch.nn.Linear(32, 2)

        self.relu = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.pool(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.fc3(self.dropout(self.relu(self.fc2(x))))
        # x = self.relu(self.fc1(x))
        # x = self.fc3(self.relu(self.fc2(x)))
        x = self.softmax(x)
        return x


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
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 1, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 1, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(115200, 256)
        self.fc2 = nn.Linear(256, num_classes)

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
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc2(self.dropout(self.relu(self.fc1(out))))
        return out


def save_model(model, name=None):
    from torch import save
    from os import path
    if name is None:
        if isinstance(model, On_Screen_Classifier):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'classifier.th'))
        elif isinstance(model, ResNet):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'classifier_resnet.th'))
        else:
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'classifier_resnet18.th'))
    else:
        if isinstance(model, On_Screen_Classifier):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '{0}.th'.format(name)))
        elif isinstance(model, ResNet):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '{0}_resnet.th'.format(name)))
        else:
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '{0}_resnet18.th'.format(name)))

    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    # TODO: Modify this function so that it would fit for different models
    from torch import load
    from os import path
    r = On_Screen_Classifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'classifier.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    from .controller import control
    from .utils import PyTux
    from argparse import ArgumentParser

    def test_planner(args):
        # Load model
        planner = load_model().eval()
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, planner=planner, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()

    parser = ArgumentParser("Test the planner")
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_planner(args)
