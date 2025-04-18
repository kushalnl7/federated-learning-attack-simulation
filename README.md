# Federated Learning: Baseline Model Evaluation

This project implements a **Federated Learning (FL)** setup using **Flower** and **PyTorch** to collaboratively train a model across multiple clients without sharing local data.

---

## Project Overview

- **Dataset**: MNIST (handwritten digits)
- **Model**: Lightweight CNN
- **Frameworks**: Flower (FL framework), PyTorch
- **Clients**: 10
- **Data Partitioning**: Mild Non-IID (8 random labels per client)
- **Rounds**: 10
- **Local Training**: 3 epochs per round
- **Optimizer**: SGD (lr = 0.05, momentum = 0.9)

---

## Results: Baseline (Clean Training)

After 10 communication rounds, the federated model achieved **~99.1% accuracy** on the MNIST test set.

### Accuracy Over Rounds

![Accuracy Curve](./global_accuracy_over_rounds.png)

---

## Next Phase: Adversarial Attack Simulation

After validating clean baseline performance, the next steps include simulating attacks such as:
- Label Flipping Attack
- Model Poisoning Attack
- Backdoor Attack

These will evaluate the **robustness** of federated learning against adversarial behaviors.
