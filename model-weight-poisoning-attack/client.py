# client_attack.py
import sys
import os

# Fix imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Fix output directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import Net

class FlowerAttackClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device, is_malicious=False, attack_type="none"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.is_malicious = is_malicious
        self.attack_type = attack_type
        if self.is_malicious and self.attack_type == "multi_label_flip":
            self.poison_map = self.generate_random_poison_map()

    def get_parameters(self, config=True):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict)

    def generate_random_poison_map(self):
        labels = np.arange(10)
        poisoned_labels = np.copy(labels)
        np.random.shuffle(poisoned_labels)
        poison_map = {original: poisoned for original, poisoned in zip(labels, poisoned_labels)}
        print(f"🔴 Malicious client poison map: {poison_map}")
        return poison_map

    def poison_labels(self, x, y):
        y_poisoned = y.clone()
        for original, poisoned in self.poison_map.items():
            y_poisoned[y == original] = poisoned
        return x, y_poisoned

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        if self.is_malicious and self.attack_type == "model_poisoning":
            print("⚠️  Malicious client sending random model parameters instead of training.")
            # Inject random noise instead of real training
            poisoned_params = [np.random.normal(0, 1, p.shape) for p in self.get_parameters()]
            return poisoned_params, len(self.train_loader.dataset), {}

        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=0.05, momentum=0.9)
        for _ in range(3):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                if self.is_malicious and self.attack_type == "multi_label_flip":
                    x, y = self.poison_labels(x, y)
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
