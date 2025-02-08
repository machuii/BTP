from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg, FedAdagrad
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset
from flwr.common import ndarrays_to_parameters, NDArrays, Scalar, Context

from typing import Union
import numpy as np
import flwr
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg


DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")

NUM_PARTITIONS = 10
BATCH_SIZE = 32


def load_datasets(partition_id: int, num_partitions: int):
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": num_partitions})
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        # Instead of passing transforms to CIFAR10(..., transform=transform)
        # we will use this function to dataset.with_transform(apply_transforms)
        # The transforms object is exactly the same
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True
    )
    valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloader, valloader, testloader


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        # 1st Block
        self.conv1_1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=5, padding=2
        )
        self.conv1_2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=5, padding=2
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, padding=1)
        self.dropout1 = nn.Dropout(p=0.2)

        # 2nd Block
        self.conv2_1 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=5, padding=2
        )
        self.conv2_2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=5, padding=2
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, padding=1)
        self.dropout2 = nn.Dropout(p=0.2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 9 * 9, 512)  # Updated input size to 4096
        self.dropout3 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # 1st Block
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        # 2nd Block
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def train(client, config, global_best, epochs):
    """Train the network on the training set."""
    acc = config["acc"]
    local_acc = config["local_acc"]
    global_acc = config["global_acc"]

    temp_model = client.net
    parameters = get_parameters(temp_model)

    local_best = client.local_best_model
    local_best_parameters = get_parameters(local_best)

    local_rand, global_rand = random.random(), random.random()

    new_weights = [None] * len(parameters)

    for index, layer in enumerate(parameters):
        new_v = acc * client.velocities[index]
        new_v = new_v + local_rand * (
            local_acc * (local_best_parameters[index] - layer)
        )
        new_v = new_v + global_rand * (global_acc * (global_best[index] - layer))
        client.velocities[index] = new_v
        new_weights[index] = layer + new_v

    set_parameters(temp_model, new_weights)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(temp_model.parameters(), lr=0.001, momentum=0.9)
    total_loss = 0.0
    for epoch in range(epochs):
        for data in client.trainloader:
            inputs, labels = data["img"].to(DEVICE), data["label"].to(DEVICE)
            optimizer.zero_grad()
            outputs = temp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    total_loss /= len(client.trainloader)

    client.net = temp_model

    if total_loss < client.local_best_loss:
        client.local_best_loss = total_loss
        client.local_best_model = client.net


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"], batch["label"]
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


class FlowerClient(NumPyClient):
    def __init__(self, partition_id, net, trainloader, valloader):
        self.partition_id = partition_id
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_best_loss = np.inf
        self.local_best_model = net
        self.velocities = [np.zeros_like(p) for p in get_parameters(net)]

    def get_parameters(self, config):
        print(f"[Client {self.partition_id}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        # and we have to send back loss instead of parameters
        if config["server_round"] % 2 == 0:
            return (
                get_parameters(self.local_best_model),
                len(self.trainloader),
                {"loss": float(self.local_best_loss)},
            )
        else:
            acc = config["acc"]
            local_acc = config["local_acc"]
            global_acc = config["global_acc"]

            temp_model = self.net
            weights = get_parameters(temp_model)

            local_best = self.local_best_model
            local_best_parameters = get_parameters(local_best)

            local_rand, global_rand = random.random(), random.random()

            new_weights = [None] * len(weights)

            for index, layer in enumerate(weights):
                new_v = acc * self.velocities[index]
                new_v = new_v + local_rand * (
                    local_acc * (local_best_parameters[index] - layer)
                )
                new_v = new_v + global_rand * (global_acc * (parameters[index] - layer))
                self.velocities[index] = new_v
                new_weights[index] = layer + new_v

            set_parameters(temp_model, new_weights)

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(temp_model.parameters(), lr=0.001, momentum=0.9)
            total_loss = 0.0
            epochs = 1
            for epoch in range(epochs):
                for data in self.trainloader:
                    inputs, labels = data["img"].to(DEVICE), data["label"].to(DEVICE)
                    optimizer.zero_grad()
                    outputs = temp_model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

            total_loss /= len(self.trainloader)

            self.net = temp_model

            if total_loss < self.local_best_loss:
                self.local_best_loss = total_loss
                self.local_best_model = self.net

            return (
                [],
                len(self.trainloader),
                {"loss": float(self.local_best_loss)},
            )

    def evaluate(self, parameters, config):
        print(f"[Client {self.partition_id}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def client_fn(context: Context) -> Client:
    net = Net().to(DEVICE)

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    trainloader, valloader, _ = load_datasets(partition_id, num_partitions)
    return FlowerClient(partition_id, net, trainloader, valloader).to_client()


# Create the ClientApp
client = ClientApp(client_fn=client_fn)


class FedPSO(flwr.server.strategy.Strategy):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.best_loss = np.inf
        self.best_client = None
        self.global_best = Net()

    def __repr__(self) -> str:
        return "FedPSO"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        net = Net()
        ndarrays = get_parameters(net)
        return ndarrays_to_parameters(ndarrays)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Create custom configs
        standard_config = {
            "lr": 0.001,
            "acc": 0.3,
            "local_acc": 0.7,
            "global_acc": 1.4,
            "server_round": server_round,
        }
        fit_configurations = []

        if server_round % 2 == 0:
            fit_configurations.append(
                (self.best_client, FitIns(parameters, standard_config))
            )
        else:
            for client in clients:
                fit_configurations.append(
                    (
                        client,
                        FitIns(parameters, standard_config),
                    )
                )
        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

        for failure in failures:
            print(f"Failure: {failure}")

        if server_round % 2 == 0:
            for client, fit_res in results:
                weights = parameters_to_ndarrays(fit_res.parameters)
                set_parameters(self.global_best, weights)
                return fit_res.parameters, {}
        else:
            for client, fit_res in results:
                loss = fit_res.metrics["loss"]
                if loss < self.best_loss:
                    self.best_loss = loss
                    self.best_client = client

        metrics_aggregated = {}

        return (
            [],
            metrics_aggregated,
        )

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        config = {"server_round": server_round}
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""

        if not results:
            print(failures, "hi")
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        accuracy_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.metrics["accuracy"])
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {"accuracy": accuracy_aggregated}
        return loss_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""

        # Let's assume we won't perform the global model evaluation on the server side.
        return None

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients


def server_fn(context: Context) -> ServerAppComponents:
    # Create FedAvg strategy
    config = ServerConfig(num_rounds=10)
    return ServerAppComponents(
        config=config,
        strategy=FedPSO(),  # <-- pass the new strategy here
    )


# Create the ServerApp
server = ServerApp(server_fn=server_fn)

# Specify the resources each of your clients need
# By default, each client will be allocated 1x CPU and 0x GPUs
backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

# When running on GPU, assign an entire GPU for each client
if DEVICE.type == "cuda":
    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}
    # Refer to our Flower framework documentation for more details about Flower simulations
    # and how to set up the `backend_config`


# Run simulation
run_simulation(
    server_app=server,
    client_app=client,
    num_supernodes=NUM_PARTITIONS,
    backend_config=backend_config,
)
