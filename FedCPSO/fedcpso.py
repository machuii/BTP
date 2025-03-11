from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import copy

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import Strategy
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from flwr.common import (
    ndarrays_to_parameters,
    NDArrays,
    Scalar,
    Context,
)

from typing import Union
import numpy as np
import flwr
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg


# DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DEVICE = torch.device("cpu")
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")

NUM_PARTITIONS = 20
BATCH_SIZE = 10

partitioner = DirichletPartitioner(
    num_partitions=NUM_PARTITIONS, alpha=0.1, partition_by="label"
)


def load_datasets(partition_id: int, num_partitions: int):
    # fds = FederatedDataset(dataset="cifar10", partitioners={"train": num_partitions})
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": partitioner})
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


# create validation set for server
server_trainloader, server_valloader, server_testloader = load_datasets(0, 1)


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
    # state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def train(net, trainloader, epochs: int, verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


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


def prune_model(model, pruning_rate):
    all_weights = []
    for p in model.parameters():
        all_weights += p.abs().view(-1)
    threshold = torch.topk(
        torch.tensor(all_weights), int(len(all_weights) * pruning_rate), largest=False
    ).values[-1]
    for p in model.parameters():
        p.data = torch.where(p.abs() < threshold, torch.tensor(0.0), p)


class FlowerClient(NumPyClient):
    def __init__(self, partition_id, net, trainloader, valloader):
        self.partition_id = partition_id
        self.trainloader = trainloader
        self.valloader = valloader
        self.client_model = net

    def get_parameters(self, config):
        return get_parameters(self.client_model)

    # this is client update
    def fit(self, parameters, config):
        set_parameters(self.client_model, parameters)
        train(self.client_model, self.trainloader, epochs=1)

        # fakequant_trainable_channel(self.client_model, 16)

        # magnitude prune self.client_model
        # params = get_params(self.client_model)
        # params = prune(params, 0.5)

        # prune_model(self.client_model, 0.5)
        params = get_parameters(self.client_model)

        # return back pruned model , accuracy on test_set
        loss, accuracy = test(self.client_model, self.valloader)
        return (
            params,
            len(self.trainloader),
            {
                "accuracy": float(accuracy),
            },
        )

    def evaluate(self, parameters, config):
        print(f"[Client {self.partition_id}] evaluate, config: {config}")
        set_parameters(self.client_model, parameters)
        loss, accuracy = test(self.client_model, self.valloader)
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


class FedCPSO(Strategy):
    def __init__(
        self,
        num_clients: int = 10,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.5,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 5,
        min_available_clients: int = 10,
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients

        self.global_model = Net().to(DEVICE)
        self.global_best_accuracy = 0
        self.global_best_model = Net().to(DEVICE)

        self.local_best_accuracy = [0] * NUM_PARTITIONS
        self.prev_client_accuracy = [0] * NUM_PARTITIONS

        self.client_best_models = [Net().to(DEVICE)] * NUM_PARTITIONS
        self.client_models = [Net().to(DEVICE)] * NUM_PARTITIONS

        self.best_neighbour_grid = [[1] * NUM_PARTITIONS] * NUM_PARTITIONS
        self.best_neighbour = [0] * NUM_PARTITIONS

        self.velocities = [
            {
                name: torch.zeros_like(param)
                for name, param in model.state_dict().items()
            }
            for model in self.client_models
        ]

    def __repr__(self) -> str:
        return "FedCPSO"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        ndarrays = get_parameters(self.global_model)
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

        standard_config = {"lr": 0.005}
        fit_configurations = []
        for idx, client in enumerate(clients):
            fit_configurations.append(
                (
                    client,
                    FitIns(
                        ndarrays_to_parameters(get_parameters(self.client_models[idx])),
                        standard_config,
                    ),
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

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        aggregated_weights = aggregate(weights_results)
        set_parameters(self.global_model, aggregated_weights)

        accuracy_list = [fit_res.metrics["accuracy"] for _, fit_res in results]
        avg_accuracy = sum(accuracy_list) / len(accuracy_list)

        # global best model
        if avg_accuracy > self.global_best_accuracy:
            self.global_best_accuracy = avg_accuracy
            set_parameters(self.global_best_model, aggregated_weights)

        # best client model
        for idx, (client, fit_res) in enumerate(results):
            if fit_res.metrics["accuracy"] > self.local_best_accuracy[idx]:
                self.local_best_accuracy[idx] = fit_res.metrics["accuracy"]
                temp = Net().to(DEVICE)
                set_parameters(temp, parameters_to_ndarrays(fit_res.parameters))
                self.client_best_models[idx] = copy.deepcopy(temp)

        # update best neighbour
        for idx, (client, fit_res) in enumerate(results):
            if fit_res.metrics["accuracy"] < self.prev_client_accuracy[idx]:
                self.best_neighbour_grid[idx][self.best_neighbour[idx]] = (
                    self.best_neighbour_grid[idx][self.best_neighbour[idx]]
                    * fit_res.metrics["accuracy"]
                )
                self.best_neighbour[idx] = self.best_neighbour_grid[idx].index(
                    max(self.best_neighbour_grid[idx])
                )

        self.prev_client_accuracy = accuracy_list

        # update client models with velocities
        for idx, (client, fit_res) in enumerate(results):
            temp_model = Net().to(DEVICE)
            set_parameters(temp_model, parameters_to_ndarrays(fit_res.parameters))
            for param_name, param in self.client_models[idx].named_parameters():
                self.velocities[idx][param_name] = 0.5 * self.velocities[idx][
                    param_name
                ] + 0.5 * (
                    (
                        self.global_best_model.state_dict()[param_name]
                        - self.client_models[idx].state_dict()[param_name]
                    )
                    + (
                        self.client_best_models[idx].state_dict()[param_name]
                        - self.client_models[idx].state_dict()[param_name]
                    )
                    + (
                        self.client_models[self.best_neighbour[idx]].state_dict()[
                            param_name
                        ]
                        - self.client_models[idx].state_dict()[param_name]
                    )
                )
                self.client_models[idx].state_dict()[param_name] = (
                    temp_model.state_dict()[param_name]
                    + self.velocities[idx][param_name]
                )

        parameters_aggregated = ndarrays_to_parameters(aggregated_weights)

        metrics_aggregated = {}
        return parameters_aggregated, metrics_aggregated

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        config = {}
        best_parameters = ndarrays_to_parameters(get_parameters(self.global_best_model))
        evaluate_ins = EvaluateIns(best_parameters, config)

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
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        accuracy_aggregated = sum(
            [evaluate_res.metrics["accuracy"] for _, evaluate_res in results]
        ) / len(results)
        metrics_aggregated = {
            "acc": accuracy_aggregated,
        }
        return loss_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""

        # Evaluate global model
        loss, acc = test(self.global_best_model, server_testloader)
        # Let's assume we won't perform the global model evaluation on the server side.
        return loss, {"accuracy": acc}

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
        strategy=FedCPSO(),  # <-- pass the new strategy here
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
