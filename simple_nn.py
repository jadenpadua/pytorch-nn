import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_IN_FEATURES: int = 4
NUM_H1_NEURONS: int = 8
NUM_H2_NEURONS: int = 9
NUM_H3_NEURONS: int = 10
NUM_OUT_CLASSES: int = 3

# Create a Model Class that inherits the base nn.Module
class Model(nn.Module):
    """
    Input Layer: (4 features of an iris flower)
    Hidden layer 1
    Hidden layer 2
    Hidden layer 3
    Output layer (3 classes of iris flower)
    """
    def __init__(self, in_features=NUM_IN_FEATURES, h1=NUM_H1_NEURONS, h2=NUM_H2_NEURONS, h3=NUM_H3_NEURONS, out_features=NUM_OUT_CLASSES):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)

        return x
# pick a manual seed for randomization
torch.manual_seed(41)
# Create instance of Model
model = Model()

