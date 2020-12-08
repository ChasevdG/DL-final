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
        self.fc3 = torch.nn.Linear(32, 1)

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
        
        return x


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, On_Screen_Classifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = Planner()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cpu'))
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
