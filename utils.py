# utils.py
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset

def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return train, test

def partition_data(train_dataset, num_clients):
    labels = np.array(train_dataset.targets)
    idx_by_label = {i: np.where(labels == i)[0].tolist() for i in range(10)}
    samples_per_client = len(train_dataset) // num_clients
    client_indices = []

    for _ in range(num_clients):
        selected = []
        chosen_labels = np.random.choice(range(10), 7, replace=False)  # 7 labels per client
        for label in chosen_labels:
            available = idx_by_label[label]
            n_samples = min(samples_per_client // 7, len(available))
            if n_samples > 0:
                sampled = np.random.choice(available, n_samples, replace=False)
                selected.extend(sampled)
                idx_by_label[label] = list(set(available) - set(sampled))
        if len(selected) == 0:
            selected = np.random.choice(np.arange(len(train_dataset)), samples_per_client, replace=False)
        client_indices.append(np.array(selected))

    return client_indices
