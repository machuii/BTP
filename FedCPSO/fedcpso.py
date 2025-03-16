from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import sys, os
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

from io import BytesIO
from typing import cast
from typing import Union
import numpy as np
import flwr
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    GetParametersIns,
    GetParametersRes,
    Status,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
    Scalar,
    Context,
    NDArrays,
    NDArray,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

import logging


logger = logging.getLogger(__name__)
logging.basicConfig(filename="D:\work\BTP\FedCPSO\logs.log", level=logging.INFO)


# DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DEVICE = torch.device("cpu")
print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")

NUM_PARTITIONS = 10
BATCH_SIZE = 10


def ndarrays_to_sparse_parameters(ndarrays: NDArrays) -> Parameters:
    """Convert NumPy ndarrays to parameters object."""
    tensors = [ndarray_to_sparse_bytes(ndarray) for ndarray in ndarrays]
    return Parameters(tensors=tensors, tensor_type="numpy.ndarray")


def sparse_parameters_to_ndarrays(parameters: Parameters) -> NDArrays:
    """Convert parameters object to NumPy ndarrays."""
    return [sparse_bytes_to_ndarray(tensor) for tensor in parameters.tensors]


def ndarray_to_sparse_bytes(ndarray: NDArray) -> bytes:
    """Serialize NumPy ndarray to bytes."""
    bytes_io = BytesIO()

    if len(ndarray.shape) > 1:
        # We convert our ndarray into a sparse matrix
        ndarray = torch.tensor(ndarray).to_sparse_csr()

        # And send it byutilizing the sparse matrix attributes
        # WARNING: NEVER set allow_pickle to true.
        # Reason: loading pickled data can execute arbitrary code
        # Source: https://numpy.org/doc/stable/reference/generated/numpy.save.html
        np.savez(
            bytes_io,  # type: ignore
            crow_indices=ndarray.crow_indices(),
            col_indices=ndarray.col_indices(),
            values=ndarray.values(),
            allow_pickle=False,
        )
    else:
        # WARNING: NEVER set allow_pickle to true.
        # Reason: loading pickled data can execute arbitrary code
        # Source: https://numpy.org/doc/stable/reference/generated/numpy.save.html
        np.save(bytes_io, ndarray, allow_pickle=False)
    return bytes_io.getvalue()


def sparse_bytes_to_ndarray(tensor: bytes) -> NDArray:
    """Deserialize NumPy ndarray from bytes."""
    bytes_io = BytesIO(tensor)
    # WARNING: NEVER set allow_pickle to true.
    # Reason: loading pickled data can execute arbitrary code
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.load.html
    loader = np.load(bytes_io, allow_pickle=False)  # type: ignore

    if "crow_indices" in loader:
        # We convert our sparse matrix back to a ndarray, using the attributes we sent
        ndarray_deserialized = (
            torch.sparse_csr_tensor(
                crow_indices=loader["crow_indices"],
                col_indices=loader["col_indices"],
                values=loader["values"],
            )
            .to_dense()
            .numpy()
        )
    else:
        ndarray_deserialized = loader
    return cast(NDArray, ndarray_deserialized)


partitioner = DirichletPartitioner(
    num_partitions=NUM_PARTITIONS, alpha=0.1, partition_by="label"
)


def load_datasets(partition_id: int):
    # fds = FederatedDataset(dataset="cifar10", partitioners={"train": num_partitions})
    fds = FederatedDataset(
        dataset="uoft-cs/cifar10", partitioners={"train": NUM_PARTITIONS}
    )
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
server_trainloader, _, server_testloader = load_datasets(0)


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
        logger.info(
            f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}, Accuracy: {epoch_acc}"
        )


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


def prune_model(net, prate):
    parameters = get_parameters(net)
    pruned_params = [None] * len(parameters)

    for idx, param in enumerate(parameters):
        if param.ndim > 1:
            flat_weights = np.abs(param).flatten()
            k = int(prate * flat_weights.size)  # Number of weights to prune
            if k > 0:
                # Find the k-th smallest magnitude
                threshold = np.partition(flat_weights, k)[k]
                # Set weights below the threshold to zero
                param[np.abs(param) < threshold] = 0
        pruned_params[idx] = param

    return pruned_params


# net = Net().to(DEVICE)
# trainloader, valloader, _ = load_datasets(0)
# train(net, trainloader, epochs=10)


class FlowerClient(Client):
    def __init__(self, partition_id, net, trainloader, valloader):
        self.partition_id = partition_id
        self.trainloader = trainloader
        self.valloader = valloader
        self.client_model = Net().to(DEVICE)
        self.model_path = f"models/client_{partition_id}.pt"

        if os.path.exists(self.model_path):
            self.client_model.load_state_dict(torch.load(self.model_path))

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:

        # Get parameters as a list of NumPy ndarray's
        ndarrays: List[np.ndarray] = get_parameters(self.client_model)

        # Serialize ndarray's into a Parameters object
        parameters = ndarrays_to_parameters(ndarrays)

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return GetParametersRes(
            status=status,
            parameters=parameters,
        )

    def fit(self, ins: FitIns) -> FitRes:

        # Deserialize parameters to NumPy ndarray's
        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)

        # Update local model, train, get updated parameters
        set_parameters(self.client_model, ndarrays_original)

        train(self.client_model, self.trainloader, epochs=2)

        # prune model
        pruned_params = prune_model(self.client_model, 0.5)
        set_parameters(self.client_model, pruned_params)

        loss, acc = test(self.client_model, self.valloader)

        # Serialize ndarray's into a Parameters object
        ndarrays_updated = get_parameters(self.client_model)
        parameters_updated = ndarrays_to_parameters(ndarrays_updated)

        torch.save(self.client_model.state_dict(), self.model_path)

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=len(self.trainloader),
            metrics={"accuracy": float(acc)},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:

        # set_parameters(self.client_model, ndarrays_original)

        loss, accuracy = test(self.client_model, self.valloader)

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=len(self.valloader),
            metrics={"accuracy": float(accuracy)},
        )


def client_fn(context: Context) -> Client:
    net = Net().to(DEVICE)

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]

    trainloader, valloader, _ = load_datasets(partition_id)
    return FlowerClient(partition_id, net, trainloader, valloader).to_client()


# Create the ClientApp
client = ClientApp(client_fn=client_fn)


### Params are not being received in the same order in each round ###
class FedCPSO(Strategy):
    def __init__(
        self,
        global_model,
        num_clients: int = 10,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.5,
        min_fit_clients: int = 1,
        min_evaluate_clients: int = 1,
        min_available_clients: int = 1,
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients

        self.global_model = global_model
        self.global_best_accuracy = 0.0
        self.global_best_model = global_model

        self.local_best_accuracy = {}
        self.prev_client_accuracy = {}

        self.client_best_models = {}
        self.client_models = {}

        self.best_neighbour_grid = {}
        self.best_neighbour = {}

        self.velocities = {}

    def __repr__(self) -> str:
        return "FedCPSO"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        clients = client_manager.all()
        weights = get_parameters(self.global_model)
        for name, client in clients.items():

            self.client_models[client.cid] = Net().to(DEVICE)
            set_parameters(self.client_models[client.cid], weights)

            self.client_best_models[client.cid] = Net().to(DEVICE)
            set_parameters(self.client_best_models[client.cid], weights)

            self.local_best_accuracy[client.cid] = 0.0
            self.prev_client_accuracy[client.cid] = 0.0

            self.velocities[client.cid] = {}
            for idx, param in enumerate(weights):
                self.velocities[client.cid][idx] = np.zeros_like(param)

            self.best_neighbour_grid[client.cid] = {}
            for neighbour_name, neighbour in clients.items():
                self.best_neighbour_grid[client.cid][neighbour.cid] = 1.0
            self.best_neighbour[client.cid] = client.cid
        return ndarrays_to_parameters(weights)

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
        for client in clients:
            fit_configurations.append(
                (
                    client,
                    FitIns(
                        ndarrays_to_parameters(
                            get_parameters(self.client_models[client.cid])
                        ),
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
        avg_acc = sum(accuracy_list) / len(accuracy_list)

        # global best model
        if avg_acc > self.global_best_accuracy:
            self.global_best_accuracy = avg_acc
            set_parameters(self.global_best_model, aggregated_weights)

        for client, fit_res in results:
            # best client model
            if fit_res.metrics["accuracy"] > self.local_best_accuracy[client.cid]:
                self.local_best_accuracy[client.cid] = fit_res.metrics["accuracy"]
                params = parameters_to_ndarrays(fit_res.parameters)
                set_parameters(self.client_best_models[client.cid], params)

            # update best neighbour
            if fit_res.metrics["accuracy"] < self.prev_client_accuracy[client.cid]:
                self.best_neighbour_grid[client.cid][
                    self.best_neighbour[client.cid]
                ] = (
                    self.best_neighbour_grid[client.cid][
                        self.best_neighbour[client.cid]
                    ]
                    * fit_res.metrics["accuracy"]
                )

                max_score = 0.0
                for neighbour, score in self.best_neighbour_grid[client.cid].items():
                    if score > max_score:
                        max_score = score
                        self.best_neighbour[client.cid] = neighbour

        for client, fit_res in results:
            self.prev_client_accuracy[client.cid] = fit_res.metrics["accuracy"]

        # update client models with velocities
        for client, fit_res in results:
            client_model_params = parameters_to_ndarrays(fit_res.parameters)
            client_best_params = get_parameters(self.client_best_models[client.cid])
            best_neighbour_params = get_parameters(
                self.client_best_models[self.best_neighbour[client.cid]]
            )
            global_best_params = get_parameters(self.global_best_model)
            new_weights = [None] * len(client_model_params)
            for idx, param in enumerate(client_model_params):
                new_v = 0.5 * (self.velocities[client.cid][idx])
                new_v += 0.5 * (global_best_params[idx] - param)
                new_v += 0.5 * (client_best_params[idx] - param)
                new_v += 0.5 * (best_neighbour_params[idx] - param)

                new_weights[idx] = param + new_v

            set_parameters(self.client_models[client.cid], new_weights)

        parameters_aggregated = None

        metrics_aggregated = {}
        return parameters_aggregated, metrics_aggregated

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        config = {}

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        send_eval = []
        for client in clients:
            evaluate_ins = EvaluateIns(
                ndarrays_to_parameters(get_parameters(self.client_models[client.cid])),
                config,
            )
            send_eval.append((client, evaluate_ins))
        # Return client/config pairs
        return send_eval

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
        loss, acc = test(self.global_model, server_testloader)

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
    config = ServerConfig(num_rounds=5)
    return ServerAppComponents(
        config=config,
        strategy=FedCPSO(
            global_model=Net(), num_clients=NUM_PARTITIONS
        ),  # <-- pass the new strategy here
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
