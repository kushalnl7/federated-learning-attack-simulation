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
train_dataset, test_dataset = load_data()
partitions = partition_data(train_dataset, num_clients=5)

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

    # Custom strategy to capture metrics
    class SaveMetricsStrategy(fl.server.strategy.FedAvg):
        def __init__(self):
            super().__init__()
            self.accuracy_history = []

        def aggregate_evaluate(self, rnd, results, failures):
            aggregated_result = super().aggregate_evaluate(rnd, results, failures)
            if results:
                accuracies = [r.metrics["accuracy"] for _, r in results if "accuracy" in r.metrics]
                if accuracies:
                    avg_acc = sum(accuracies) / len(accuracies)
                    self.accuracy_history.append((rnd, avg_acc))
                    print(f"📈 [Round {rnd}] Average Accuracy: {avg_acc * 100:.2f}%")
            return aggregated_result

    strategy = SaveMetricsStrategy()

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=5,
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )

    end_time = time.time()
    print(f"\n⏱️ Simulation completed in {end_time - start_time:.2f} seconds.")

    # Save per-round results to CSV
    with open("round_accuracy_results.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Round", "Accuracy"])
        for round_num, acc in strategy.accuracy_history:
            writer.writerow([round_num, acc])

    print("\n📁 Saved round-wise accuracies to 'round_accuracy_results.csv'.")

    # Plot Accuracy over Rounds
    if strategy.accuracy_history:
        rounds = [r for r, _ in strategy.accuracy_history]
        accuracies = [a for _, a in strategy.accuracy_history]

        plt.plot(rounds, accuracies, marker="o")
        plt.title("Federated Learning: Accuracy over Rounds")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.savefig("accuracy_over_rounds.png")
        print("\n📈 Saved plot to 'accuracy_over_rounds.png'.")
    else:
        print("⚠️ No round-wise accuracy data available to plot.")

    # Evaluate final model centrally (optional: after aggregation)
    print("\n📊 Evaluating final model centrally on test data...")
    model = Net().to(DEVICE)
    test_loader = DataLoader(test_dataset, batch_size=64)

    model.eval()
    loss, correct = 0.0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            loss += torch.nn.functional.cross_entropy(pred, y, reduction="sum").item()
            correct += (pred.argmax(1) == y).sum().item()

    final_accuracy = correct / len(test_loader.dataset)
    print(f"\n✅ Final global test accuracy: {final_accuracy * 100:.2f}%")

    with open("results_summary.txt", "w") as f:
        f.write(f"Final Accuracy: {final_accuracy * 100:.2f}%\n")
        f.write(f"Training Time: {end_time - start_time:.2f} seconds\n")

    print("\n📁 Summary saved to 'results_summary.txt'.")
