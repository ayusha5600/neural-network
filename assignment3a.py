import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics, model_selection
from sklearn.utils import shuffle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = r"C:\Users\Administrator\Documents\semister 4\Artificial inteligence\classification\dataset_014.csv"


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_data():
    df = pd.read_csv(DATASET_PATH)
    X = df.iloc[:, :-1].values
    y_raw = df.iloc[:, -1].values

    labels_uni = np.unique(y_raw)
    label2idx = {label: idx for idx, label in enumerate(labels_uni)}
    y = np.array([label2idx[v] for v in y_raw], dtype=np.int64)

    scaler = sklearn.preprocessing.StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return (
        np.array(X_train, dtype=np.float32),
        np.array(X_test, dtype=np.float32),
        np.array(y_train, dtype=np.int64),
        np.array(y_test, dtype=np.int64),
        X.shape[1],
        len(np.unique(y)),
        df.shape,
    )


class DeepMLP(nn.Module):
    def __init__(self, input_size: int, num_classes: int, hidden_sizes, dropout: float):
        super().__init__()
        layers = []
        prev = input_size

        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_one_config(cfg, X_train, y_train, X_test, y_test, input_size, num_classes, device):
    model = DeepMLP(
        input_size=input_size,
        num_classes=num_classes,
        hidden_sizes=cfg["hidden_sizes"],
        dropout=cfg["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    if cfg["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    batch_size = cfg["batch_size"]
    total_steps = len(y_train) // batch_size
    losses = []

    model.train()
    for epoch in range(cfg["num_epochs"]):
        x_s, y_s = shuffle(X_train, y_train)
        for step in range(total_steps):
            xb = x_s[step * batch_size : (step + 1) * batch_size]
            yb = y_s[step * batch_size : (step + 1) * batch_size]

            xb_t = torch.from_numpy(xb).to(device)
            yb_t = torch.from_numpy(yb).to(device)

            logits = model(xb_t)
            loss = criterion(logits, yb_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        if (epoch + 1) % 50 == 0:
            print(f"[{cfg['name']}] Epoch {epoch+1}/{cfg['num_epochs']} Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        x_test_t = torch.from_numpy(X_test).to(device)
        logits = model(x_test_t)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        y_pred = np.argmax(probs, axis=1)

    result = {
        "name": cfg["name"],
        "accuracy": metrics.accuracy_score(y_test, y_pred),
        "precision": metrics.precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": metrics.recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": metrics.f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "roc_auc": metrics.roc_auc_score(y_test, probs, multi_class="ovr", average="weighted"),
        "confusion_matrix": metrics.confusion_matrix(y_test, y_pred),
        "classification_report": metrics.classification_report(y_test, y_pred, zero_division=0),
        "losses": losses,
    }

    return model, result


def main():
    print("=" * 80)
    print("ASSIGNMENT 3A (Teacher Style) - assignment3a.py")
    print("=" * 80)

    X_train, X_test, y_train, y_test, input_size, num_classes, data_shape = load_data()
    device = get_device()

    print(f"Dataset shape: {data_shape}")
    print(f"Input size: {input_size}, Classes: {num_classes}, Device: {device}")

    configs = [
        {
            "name": "Run 1",
            "hidden_sizes": [64, 32],
            "dropout": 0.0,
            "num_epochs": 250,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "AdamW",
        },
        {
            "name": "Run 2",
            "hidden_sizes": [128, 64, 32],
            "dropout": 0.1,
            "num_epochs": 350,
            "batch_size": 16,
            "learning_rate": 0.001,
            "optimizer": "Adam",
        },
        {
            "name": "Run 3",
            "hidden_sizes": [256, 128, 64, 32],
            "dropout": 0.2,
            "num_epochs": 400,
            "batch_size": 16,
            "learning_rate": 0.0007,
            "optimizer": "AdamW",
        },
    ]

    all_results = []
    model_paths = []

    for i, cfg in enumerate(configs, start=1):
        print("\n" + "-" * 80)
        print(
            f"Training {cfg['name']} with hidden={cfg['hidden_sizes']}, "
            f"dropout={cfg['dropout']}, lr={cfg['learning_rate']}, batch={cfg['batch_size']}"
        )
        model, result = train_one_config(cfg, X_train, y_train, X_test, y_test, input_size, num_classes, device)
        all_results.append(result)

        model_path = os.path.join(BASE_DIR, f"model_run_{i}.ckpt")
        torch.save(model.state_dict(), model_path)
        model_paths.append(model_path)
        print(f"Saved weights: {model_path}")
        print(f"Accuracy: {result['accuracy']:.4f} | F1: {result['f1']:.4f}")

    best_idx = int(np.argmax([r["accuracy"] for r in all_results]))
    best_weights_path = model_paths[best_idx]
    best_weights_copy_path = os.path.join(BASE_DIR, "trained_model_weights.ckpt")
    with open(best_weights_path, "rb") as src, open(best_weights_copy_path, "wb") as dst:
        dst.write(src.read())

    report_path = os.path.join(BASE_DIR, "report_assignment3a.txt")
    with open(__file__, "r", encoding="utf-8") as code_file:
        code_text = code_file.read()

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("ASSIGNMENT 3A REPORT\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Dataset: {DATASET_PATH}\n")
        f.write(f"Data shape: {data_shape}\n\n")

        f.write("USED PYTHON CODE\n")
        f.write("-" * 80 + "\n")
        f.write(code_text + "\n\n")

        f.write("RESULTS OF CLASSIFICATION\n")
        f.write("-" * 80 + "\n")

        for idx, res in enumerate(all_results, start=1):
            f.write(f"{res['name']}\n")
            f.write(f"Accuracy: {res['accuracy']:.4f}\n")
            f.write(f"Precision: {res['precision']:.4f}\n")
            f.write(f"Recall: {res['recall']:.4f}\n")
            f.write(f"F1: {res['f1']:.4f}\n")
            f.write(f"ROC-AUC: {res['roc_auc']:.4f}\n")
            f.write("Classification report:\n")
            f.write(res["classification_report"] + "\n")
            f.write("Confusion matrix:\n")
            f.write(str(res["confusion_matrix"]) + "\n\n")

        f.write(f"Best run: {all_results[best_idx]['name']}\n")

        f.write("EXPLANATION OF RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write("1. Deeper architecture and dropout change generalization behavior across runs.\n")
        f.write("2. Weighted metrics summarize class-wise quality in a single score set.\n")
        f.write("3. ROC-AUC (OVR) is included for multi-class discrimination quality.\n\n")

        f.write("CONCLUSIONS\n")
        f.write("-" * 80 + "\n")
        f.write("1. Three runs were repeated with changed architecture/hyperparameters.\n")
        f.write("2. assignment3a.py uses deeper MLP than assignment3.py.\n")
        f.write("3. Weights were saved for every run as required.\n")
        f.write(f"4. Best model weights file: {best_weights_copy_path}\n")

    plt.figure(figsize=(10, 5))
    for i, res in enumerate(all_results, start=1):
        plt.plot(res["losses"], label=f"Run {i}")
    plt.title("Training Loss Curves - assignment3a")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(BASE_DIR, "loss_assignment3a.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print("\nDone")
    print(f"Report: {report_path}")
    print(f"Loss plot: {plot_path}")
    print(f"Best weights copy: {best_weights_copy_path}")


if __name__ == "__main__":
    main()
