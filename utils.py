# utils_clean.py
from torchvision import datasets, transforms
import numpy as np

def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return train, test

def partition_data(train_dataset, num_clients, non_iid=True):
    data_per_client = len(train_dataset) // num_clients
    labels = np.array(train_dataset.targets)

    if not non_iid:
        # IID
        all_indices = np.arange(len(train_dataset))
        np.random.shuffle(all_indices)
        return [
            all_indices[i * data_per_client : (i + 1) * data_per_client]
            for i in range(num_clients)
        ]

    # Balanced Non-IID Partition
    idx_by_label = {i: np.where(labels == i)[0].tolist() for i in range(10)}
    client_indices = []

    for _ in range(num_clients):
        chosen_labels = np.random.choice(range(10), 8, replace=False)
        selected_indices = []
        for label in chosen_labels:
            available = idx_by_label[label]
            sample_size = min(data_per_client // 8, len(available))
            if sample_size > 0:
                sampled = np.random.choice(available, sample_size, replace=False)
                selected_indices.extend(sampled)
                idx_by_label[label] = list(set(available) - set(sampled))
        if len(selected_indices) == 0:
            fallback = np.arange(len(train_dataset))
            selected_indices = np.random.choice(fallback, data_per_client, replace=False)
        client_indices.append(np.array(selected_indices))
    return client_indices
