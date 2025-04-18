# app.py
import torch
from torch.utils.data import DataLoader, Subset
import flwr as fl
import matplotlib.pyplot as plt
import time
import csv
from model import Net
from utils import load_data, partition_data
from client import FlowerClient

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Centralized evaluation function
def evaluate_global_model(model, test_loader):
    model.eval()
    loss, correct = 0.0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            loss += torch.nn.functional.cross_entropy(pred, y, reduction="sum").item()
            correct += (pred.argmax(1) == y).sum().item()
    accuracy = correct / len(test_loader.dataset)
    return accuracy

train_dataset, test_dataset = load_data()
partitions = partition_data(train_dataset, num_clients=10)

def client_fn(cid: str):
    model = Net()
    cid_int = int(cid)
    client_data = Subset(train_dataset, partitions[cid_int])
    train_loader = DataLoader(client_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    return FlowerClient(model, train_loader, test_loader, DEVICE)

if __name__ == "__main__":
    print("\n🚀 Starting Federated Learning Simulation...")

    start_time = time.time()

    global_model = Net().to(DEVICE)
    test_loader = DataLoader(test_dataset, batch_size=64)

    class CentralizedEvalStrategy(fl.server.strategy.FedAvg):
        def __init__(self):
            super().__init__()
            self.global_model = global_model
            self.history = []

        def aggregate_fit(self, rnd, results, failures):
            aggregated_result = super().aggregate_fit(rnd, results, failures)

            if aggregated_result is not None:
                parameters_aggregated, _ = aggregated_result
                params_ndarrays = fl.common.parameters_to_ndarrays(parameters_aggregated)
                params_dict = zip(self.global_model.state_dict().keys(), params_ndarrays)
                state_dict = {k: torch.tensor(v) for k, v in params_dict}
                self.global_model.load_state_dict(state_dict, strict=True)

            print(f"\n🧹 [Round {rnd}] Aggregating and Evaluating on Full Test Set...")
            acc = evaluate_global_model(self.global_model, test_loader)
            print(f"📈 [Round {rnd}] Global Test Accuracy: {acc*100:.2f}%")
            self.history.append((rnd, acc))

            return aggregated_result

    strategy = CentralizedEvalStrategy()

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=10,
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )

    end_time = time.time()
    print(f"\n⏱️ Simulation completed in {end_time - start_time:.2f} seconds.")

    # Save round-wise results
    with open("round_centralized_accuracy.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Round", "Accuracy"])
        writer.writerows(strategy.history)

    # Plot
    rounds = [r for r, _ in strategy.history]
    accuracies = [a for _, a in strategy.history]

    plt.plot(rounds, accuracies, marker="o")
    plt.title("Global Test Accuracy over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig("global_accuracy_over_rounds.png")
    # plt.show()

    print("\n✅ Results saved successfully!")
