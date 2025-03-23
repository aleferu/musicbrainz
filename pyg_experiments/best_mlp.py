#!/usr/bin/env python3


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, random_split
import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix


class MLP(nn.Module):
    def __init__(self, input_size: int, layers: int, hl_size: int, lr: float, wd: float):
        super(MLP, self).__init__()
        self.name = f"{layers}-{hl_size}-{lr}-{wd}"
        self.probs = list()
        self.labels = list()
        self.preds = list()
        self.best_f1: float = -1
        self.best_roc_auc: float = -1

        self.layers = nn.Sequential()
        self.layers.append(nn.Linear(input_size, hl_size))
        self.layers.append(nn.ReLU())

        for l in range(layers - 1):
            self.layers.append(nn.Linear(hl_size, hl_size))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(hl_size, 1))

        self.optimizer = optim.AdamW(self.layers.parameters(), lr=lr, weight_decay=wd)

    def forward(self, x):
        return self.layers(x)


def main():
    x_train = torch.load(f"pyg_experiments/ds/mlp2023110x_train.pt")
    x_test = torch.load(f"pyg_experiments/ds/mlp2023110x_test.pt")
    y_train = torch.load(f"pyg_experiments/ds/mlp2023110y_train.pt")
    y_test = torch.load(f"pyg_experiments/ds/mlp2023110y_test.pt")

    input_size = x_train.shape[1]

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=10, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, num_workers=10, pin_memory=True)

    eslr = [1e-3, 1e-4]
    eswd = [1e-2, 1e-3, 1e-4, 0.0]
    models = [MLP(input_size, layers, hl_size, lr, wd).to(device) for layers in [1, 3] for hl_size in [128, 256] for lr in eslr for wd in eswd]
    print(len(models))
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1000):
        print(f"Epoch {epoch + 1}")
        for model in models:
            model.train()

        print("Training...")
        for inputs, targets in tqdm.tqdm(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            for model in models:
                model.optimizer.zero_grad()
                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, targets)
                loss.backward()
                model.optimizer.step()

        print("Evaluating...")
        for model in models:
            model.eval()
        with torch.no_grad():
            for inputs, targets in tqdm.tqdm(test_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)

                for model in models:
                    outputs = model(inputs).squeeze(1)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    preds = (probs > 0.5).astype(int)
                    labels = targets.cpu().numpy()
                    model.probs.extend(probs.flatten())
                    model.labels.extend(labels)
                    model.preds.extend(preds.flatten())

        print("F1?")
        for model in models:
            cm = confusion_matrix(model.labels, model.preds)
            tp = cm[1, 1]
            fp = cm[0, 1]
            fn = cm[1, 0]
            tn = cm[0, 0]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            roc_auc = roc_auc_score(model.labels, model.probs)
            if f1 > model.best_f1:
                model.best_f1 = f1
            if roc_auc > model.best_roc_auc:
                model.best_roc_auc = float(roc_auc)
            model.labels = list()
            model.preds = list()
            model.probs = list()

        models = sorted(models, key=lambda m: m.best_f1)

        for i, model in enumerate(models):
            print(f"{i+1}:")
            print(f"   {model.name}: {model.best_f1:.4f}")
            print(f"   {model.name}: {model.best_roc_auc:.4f}")

        print()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: '{device}'")
    main()
