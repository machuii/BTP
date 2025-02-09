import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = Net()

# prune.l1_unstructured(model.fc1, name="weight", amount=0.5)
# prune.l1_unstructured(model.fc2, name="weight", amount=0.5)


# prune.remove(model.fc1, "weight")
# prune.remove(model.fc2, "weight")


## magnitude prune the model


def prune_model(model, pruning_rate):
    all_weights = []
    for p in model.parameters():
        all_weights += p.abs().view(-1)
    threshold = torch.topk(
        torch.tensor(all_weights), int(len(all_weights) * pruning_rate), largest=False
    ).values[-1]
    for p in model.parameters():
        p.data = torch.where(p.abs() < threshold, torch.tensor(0.0), p)


for name, param in model.named_parameters():
    print(f"{name}: {torch.sum(param == 0).item()} weights pruned")

prune_model(model, 0.5)


for name, param in model.named_parameters():
    print(f"{name}: {torch.sum(param == 0).item()} weights pruned")
