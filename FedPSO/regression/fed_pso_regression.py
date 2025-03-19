from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import random

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.simulation import run_simulation
from flwr.common import (
    GetParametersIns,
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
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    ParametersRecord,
    ConfigsRecord,
    array_from_numpy,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")

NUM_PARTITIONS = 10
BATCH_SIZE = 32


# need a regression dataset like housing or air quality
def prepare_dataset():
    data = fetch_california_housing(as_frame=True)
    X, y = data.data, data.target

    X = (X.mean() - X) / X.std()

    client_data = []
    for i in range(NUM_PARTITIONS):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        test_loader = DataLoader(
            TensorDataset(X_test_tensor, y_test_tensor),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        client_data.append((train_loader, test_loader))

    return client_data


housing_data = prepare_dataset()


def load_datasets(partition_id):
    trainloader, valloader = housing_data[partition_id]
    return trainloader, valloader


# need a regressor model
class Net(nn.Module):
    def __init__(self, input_size=8):
        super(Net, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_size, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        x = self.fc(x)
        return x


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def train(net, trainloader, epochs: int):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        epoch_loss, total = 0.0, 0
        for batch in trainloader:
            images, labels = batch
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # metrics
            epoch_loss += loss.item()
            total += labels.size(0)
        epoch_loss /= len(trainloader.dataset)
        # print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}, Accuracy: {epoch_acc}")
    return epoch_loss


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.MSELoss()
    loss = 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
    loss /= len(testloader.dataset)
    return loss


class FlowerClient(NumPyClient):
    def __init__(self, context: Context, trainloader, valloader):

        self.client_state = context.state
        self.partition_id = context.node_config["partition-id"]

        if "loss" not in self.client_state.configs_records:
            self.client_state.configs_records["loss"] = ConfigsRecord()
            self.client_state.configs_records["loss"]["global_best_loss"] = 0.0
            self.client_state.configs_records["loss"]["local_best_loss"] = 0.0

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

        loss = self.client_state.configs_records["loss"]
        local_best_loss = loss["local_best_loss"]
        global_best_loss = loss["global_best_loss"]

        if global_best_loss < config["global_best_loss"]:
            set_parameters(self.global_best_model, parameters)
            global_best_loss = config["global_best_loss"]

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

        train_loss = train(temp_model, self.trainloader, epochs=2)

        trained_weights = get_parameters(temp_model)

        set_parameters(self.net, trained_weights)

        if train_loss >= local_best_loss:
            local_best_loss = train_loss
            set_parameters(self.local_best_model, trained_weights)

        loss["local_best_loss"] = local_best_loss
        loss["global_best_loss"] = global_best_loss

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
            {"loss": float(train_loss)},
        )

    def evaluate(self, parameters, config):
        test_loss = test(self.net, self.valloader)
        return float(test_loss), len(self.valloader), {}


def client_fn(context: Context) -> Client:
    partition_id = context.node_config["partition-id"]
    trainloader, valloader = load_datasets(partition_id)
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
        self.best_loss = 0.0
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
            "global_best_loss": self.best_loss,
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
            if fit_res.metrics["loss"] > self.best_loss:
                self.best_loss = fit_res.metrics["loss"]
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

        return loss_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""
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
    config = ServerConfig(num_rounds=10)
    return ServerAppComponents(
        config=config,
        strategy=FedPSO(fraction_fit=0.5),  # <-- pass the new strategy here
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
