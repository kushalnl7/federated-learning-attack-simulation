# app_attack.py
import sys
import os
from client import FlowerBackdoorClient

# Fix imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Fix output directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch  # type: ignore
from torch.utils.data import DataLoader, Subset # type: ignore
import flwr as fl # type: ignore
import matplotlib.pyplot as plt # type: ignore
import random
import csv
import time
from model import Net
from utils import load_data, partition_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset, test_dataset = load_data()
partitions = partition_data(train_dataset, num_clients=10)

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

def run_simulation(malicious_percentage):
    print(f"\n🚀 Running simulation with {malicious_percentage}% malicious clients...")

    num_clients = 10
    num_malicious = int(malicious_percentage / 100 * num_clients)
    malicious_cids = random.sample(range(num_clients), num_malicious)

    def client_fn(cid: str):
        model = Net()
        cid_int = int(cid)
        client_data = Subset(train_dataset, partitions[cid_int])
        train_loader = DataLoader(client_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64)
        is_malicious = cid_int in malicious_cids
        return FlowerBackdoorClient(model, train_loader, test_loader, DEVICE, is_malicious)

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

            acc = evaluate_global_model(self.global_model, test_loader)
            print(f"📈 [Round {rnd}] Global Test Accuracy: {acc*100:.2f}%")
            self.history.append((rnd, acc))

            return aggregated_result

    start_time = time.time()

    strategy = CentralizedEvalStrategy()

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
        ray_init_args={"runtime_env": {"working_dir": ".."}}
    )

    end_time = time.time()
    print(f"⏱️ Simulation with {malicious_percentage}% malicious clients completed in {end_time - start_time:.2f} seconds.")

    return strategy.history

if __name__ == "__main__":
    attack_percentages = [10, 20, 50]
    final_results = []

    for attack in attack_percentages:
        history = run_simulation(attack)
        final_accuracy = history[-1][1]
        final_results.append((attack, final_accuracy))

    with open("results_backdoor.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Malicious_Percentage", "Final_Accuracy"])
        writer.writerows(final_results)

    attack_levels = [a for a, _ in final_results]
    accuracies = [a for _, a in final_results]

    plt.plot(attack_levels, accuracies, marker="o")
    plt.title("Model Robustness under Backdoor Attack")
    plt.xlabel("Malicious Client Percentage")
    plt.ylabel("Final Global Test Accuracy")
    plt.grid(True)
    plt.savefig("accuracy_under_backdoor_attack.png")
    # plt.show()

    print("\n✅ Backdoor attack simulation results saved successfully!")