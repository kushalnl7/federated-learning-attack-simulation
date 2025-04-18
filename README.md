# Federated Learning: Baseline Model Evaluation

This project implements a **Federated Learning (FL)** setup using **Flower** and **PyTorch** to collaboratively train a model across multiple clients without sharing local data.

---

## Project Overview

- **Dataset**: MNIST (handwritten digits)
- **Model**: Lightweight CNN
- **Frameworks**: Flower (FL framework), PyTorch
- **Clients**: 5
- **Data Partitioning**: Mild Non-IID (5 random labels per client)
- **Rounds**: 10
- **Local Training**: 3 epochs per round
- **Optimizer**: SGD (lr = 0.05, momentum = 0.9)

---

## Results: Baseline (Clean Training)

After 10 communication rounds, the federated model achieved **~98.8% accuracy** on the MNIST test set.

### Accuracy Over Rounds

![Accuracy Curve](./6bf6b6cf-0039-4b9f-8abe-c38aadc8639a.png)

| Round | Accuracy |
|:------|:---------|
| 1     | 80.92%   |
| 2     | 89.74%   |
| 3     | 95.07%   |
| 4     | 97.32%   |
| 5     | 97.70%   |
| 6     | 97.93%   |
| 7     | 98.43%   |
| 8     | 98.51%   |
| 9     | 98.70%   |
| 10    | 98.82%   |

---

## Next Phase: Adversarial Attack Simulation

After validating clean baseline performance, the next steps include simulating attacks such as:
- Label Flipping Attack
- Model Poisoning Attack
- Backdoor Attack

These will evaluate the **robustness** of federated learning against adversarial behaviors.
