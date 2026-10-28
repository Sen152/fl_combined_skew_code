"""Flower客户端实现"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import flwr as fl
import numpy as np

from src.models.cnn import get_model, get_model_params, set_model_params


class FLClient(fl.client.NumPyClient):
    """联邦学习客户端"""

    def __init__(self, dataset_name: str, trainloader: DataLoader, testloader: DataLoader,
                 device: str = "cuda", local_epochs: int = 2, lr: float = 0.01):
        self.dataset_name = dataset_name
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.local_epochs = local_epochs
        self.lr = lr
        self.model = get_model(dataset_name, device)

    def get_parameters(self, config=None):
        return get_model_params(self.model)

    def fit(self, parameters, config):
        set_model_params(self.model, parameters)
        self.model.train()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        for _ in range(self.local_epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(images), labels)
                loss.backward()
                optimizer.step()

        return get_model_params(self.model), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_model_params(self.model, parameters)
        self.model.eval()

        criterion = nn.CrossEntropyLoss()
        correct, total, loss_sum = 0, 0, 0.0

        with torch.no_grad():
            for images, labels in self.testloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss_sum += criterion(outputs, labels).item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        avg_loss = loss_sum / total
        return avg_loss, total, {"accuracy": accuracy}
