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

class FlowerBackdoorClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device, is_malicious=False):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.is_malicious = is_malicious

    def get_parameters(self, config=True):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict)

    def add_backdoor_trigger(self, x_batch):
        # x_batch: [batch_size, 1, 28, 28] (MNIST)
        x_triggered = x_batch.clone()
        # Add white square trigger at bottom right 4x4 pixels
        x_triggered[:, :, 24:28, 24:28] = 1.0
        return x_triggered

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=0.05, momentum=0.9)

        for _ in range(3):  # 3 local epochs
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)

                if self.is_malicious:
                    # Poison 30% of batch samples
                    batch_size = x.size(0)
                    n_trigger = int(0.3 * batch_size)
                    idx = torch.randperm(batch_size)[:n_trigger]
                    x[idx] = self.add_backdoor_trigger(x[idx])
                    y[idx] = torch.zeros_like(y[idx])  # Force label to 0 (target class)

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