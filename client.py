# client_clean.py
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from model import Net

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def get_parameters(self, config=True):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=0.05, momentum=0.9)
        for _ in range(3):
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
