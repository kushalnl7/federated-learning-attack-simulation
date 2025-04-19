# client_attack.py
import sys
import os

# Fix imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Fix output directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import flwr as fl # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import numpy as np # type: ignore
from model import Net

class FlowerFreeRiderClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device, is_malicious=False):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.is_malicious = is_malicious
        self.last_received_parameters = None  # Save received parameters

    def get_parameters(self, config=True):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.last_received_parameters = parameters

        if self.is_malicious:
            print(f"⚠️ Free-rider client sending back received parameters without any local training!")
            # Simply return the parameters received from server
            return self.last_received_parameters, len(self.train_loader.dataset), {}

        # Otherwise train normally
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=0.05, momentum=0.9)

        for _ in range(3):  # 3 local epochs
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = nn.CrossEntropyLoss()(self.model(x), y)
                loss.backward()
                optimizer.step()

        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss, correct = 0.0, 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss += nn.CrossEntropyLoss()(pred, y).item()
                correct += (pred.argmax(1) == y).sum().item()
        accuracy = correct / len(self.test_loader.dataset)
        return float(loss), len(self.test_loader.dataset), {"accuracy": accuracy}
