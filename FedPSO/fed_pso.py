from collections import OrderedDict, Counter
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random
import copy
import os, shutil
import sys

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

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
    NDArrays,
    Context,
    GetParametersIns,
    ParametersRecord,
    ConfigsRecord,
    array_from_numpy,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

import logging

logging.basicConfig(filename="D:\work\BTP\FedPSO\logs.log", level=logging.INFO)


DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")


NUM_PARTITIONS = 10
NUM_ROUNDS = 5
BATCH_SIZE = 10

# remember to delete previous models

partitioner = IidPartitioner(num_partitions=NUM_PARTITIONS)


def load_datasets(partition_id: int):
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_PARTITIONS})
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(
        test_size=0.2, shuffle=True, seed=42
    )
    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        # Instead of passing transforms to CIFAR10(..., transform=transform)
        # we will use this function to dataset.with_transform(apply_transforms)
        # The transforms object is exactly the same
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    # Create train/val for each partition and wrap it into DataLoader
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True
    )
    valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloader, valloader, testloader


trainloader, valloader, testloader = load_datasets(1)


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def train(net, trainloader, epochs: int):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch["img"], batch["label"]
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # metrics
            epoch_loss += loss.item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        # print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}, Accuracy: {epoch_acc}")
    return epoch_loss, epoch_acc


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
            _, predicted = torch.max(outputs, 1)
            loss += criterion(outputs, labels).item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


class FlowerClient(NumPyClient):
    def __init__(self, context: Context, trainloader, valloader):

        self.client_state = context.state
        self.partition_id = context.node_config["partition-id"]

        if "accuracy" not in self.client_state.configs_records:
            self.client_state.configs_records["accuracy"] = ConfigsRecord()
            self.client_state.configs_records["accuracy"]["global_best_acc"] = 0.0
            self.client_state.configs_records["accuracy"]["local_best_acc"] = 0.0

        self.trainloader = trainloader
        self.valloader = valloader

        if "velocities" not in self.client_state.parameters_records:
            velocities = [np.zeros_like(p) for p in get_parameters(Net())]
            vel_record = ParametersRecord()
            for i in range(len(velocities)):
                vel_record[str(i)] = array_from_numpy(velocities[i])
            self.client_state.parameters_records["velocities"] = vel_record

        self.model_path = f"models/client_{self.partition_id}.pth"
        self.local_best_path = f"models/local_best_{self.partition_id}.pth"
        self.global_best_path = f"models/global_best_{self.partition_id}.pth"

        self.net = Net().to(DEVICE)
        self.local_best_model = Net().to(DEVICE)
        self.global_best_model = Net().to(DEVICE)

        if os.path.exists(self.model_path):
            self.net.load_state_dict(torch.load(self.model_path))

        if os.path.exists(self.local_best_path):
            self.local_best_model.load_state_dict(torch.load(self.local_best_path))

        if os.path.exists(self.global_best_path):
            self.global_best_model.load_state_dict(torch.load(self.global_best_path))

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        vel_record = self.client_state.parameters_records["velocities"]
        velocities = []
        for key, value in vel_record.items():
            velocities.append(value.numpy())

        accuracy = self.client_state.configs_records["accuracy"]
        local_best_acc = accuracy["local_best_acc"]
        global_best_acc = accuracy["global_best_acc"]

        if global_best_acc < config["global_best_acc"]:
            set_parameters(self.global_best_model, parameters)
            global_best_acc = config["global_best_acc"]

        acce = config["acc"]
        local_acce = config["local_acc"]
        global_acce = config["global_acc"]

        temp_model = Net().to(DEVICE)
        client_parameters = get_parameters(self.net)
        set_parameters(temp_model, client_parameters)

        client_best_parameters = get_parameters(self.local_best_model)
        global_best_parameters = get_parameters(self.global_best_model)

        new_weights = [None] * len(client_parameters)
        local_rand, global_rand = random.random(), random.random()

        for index, layer in enumerate(client_parameters):
            new_v = acce * velocities[index]
            new_v = new_v + local_rand * (
                local_acce * (client_best_parameters[index] - layer)
            )
            new_v = new_v + global_rand * (
                global_acce * (global_best_parameters[index] - layer)
            )
            velocities[index] = new_v
            new_weights[index] = layer + new_v

        set_parameters(temp_model, new_weights)

        loss, acc = train(temp_model, self.trainloader, epochs=1)

        trained_weights = get_parameters(temp_model)

        set_parameters(self.net, trained_weights)

        if acc >= local_best_acc:
            local_best_acc = acc
            set_parameters(self.local_best_model, trained_weights)

        accuracy["local_best_acc"] = local_best_acc
        accuracy["global_best_acc"] = global_best_acc

        torch.save(self.net.state_dict(), self.model_path)
        torch.save(self.local_best_model.state_dict(), self.local_best_path)
        torch.save(self.global_best_model.state_dict(), self.global_best_path)

        vel_record = ParametersRecord()
        for i in range(len(velocities)):
            vel_record[str(i)] = array_from_numpy(velocities[i])
        self.client_state.parameters_records["velocities"] = vel_record

        return (
            [],
            len(self.trainloader),
            {"acc": float(acc)},
        )

    def evaluate(self, parameters, config):
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def client_fn(context: Context) -> Client:
    partition_id = context.node_config["partition-id"]
    trainloader, valloader, _ = load_datasets(partition_id)
    return FlowerClient(context, trainloader, valloader).to_client()


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
        self.best_accuracy = 0.0
        self.best_client = None
        self.server_model = Net().to(DEVICE)

    def __repr__(self) -> str:
        return "FedPSO"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        return ndarrays_to_parameters(get_parameters(self.server_model))

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
        params = ndarrays_to_parameters(get_parameters(self.server_model))
        # Create custom configs
        standard_config = {
            "lr": 0.001,
            "acc": 0.3,
            "local_acc": 0.7,
            "global_acc": 1.4,
            "global_best_acc": self.best_accuracy,
        }
        fit_configurations = []
        for client in clients:
            fit_configurations.append(
                (
                    client,
                    FitIns(params, standard_config),
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

        for client, fit_res in results:
            if fit_res.metrics["acc"] > self.best_accuracy:
                self.best_accuracy = fit_res.metrics["acc"]
                self.best_client = client
                print(client.cid)

        global_best_parameters = self.best_client.get_parameters(
            (GetParametersIns(config={})), timeout=None, group_id=None
        ).parameters

        global_best_ndarray = parameters_to_ndarrays(global_best_parameters)
        set_parameters(self.server_model, global_best_ndarray)

        metrics_aggregated = {}

        return global_best_parameters, metrics_aggregated

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        config = {}

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

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        metrics_aggregated = {}

        accuracy_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.metrics["accuracy"])
                for _, evaluate_res in results
            ]
        )

        metrics_aggregated["accuracy"] = accuracy_aggregated
        return loss_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""
        loss, accuracy = test(self.server_model, valloader)
        print(f"Round {server_round}, Loss: {loss}, Accuracy: {accuracy}")
        # # Let's assume we won't perform the global model evaluation on the server side.
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
    config = ServerConfig(num_rounds=NUM_ROUNDS)
    return ServerAppComponents(
        config=config,
        strategy=FedPSO(fraction_fit=1.0),  # <-- pass the new strategy here
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

# Cleanup
folder_path = "models"

if os.path.exists(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    print("All files deleted successfully.")
else:
    print("Folder not found.")
