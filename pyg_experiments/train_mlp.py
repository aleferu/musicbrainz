#!/usr/bin/env python3


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, random_split
import tqdm
import copy
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_curve
import requests


class MLP(nn.Module):
    def __init__(self, input_size: int):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.layers(x)


class miniMLP(nn.Module):
    def __init__(self, input_size: int):
        super(miniMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.layers(x)


def calculate_metrics(threshold, all_probs, all_labels):
    preds_binary = (all_probs > threshold).astype(int)
    cm = confusion_matrix(all_labels, preds_binary)
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tn = cm[0, 0]
    precision = 0 if tp == 0 else tp / (tp + fp)
    recall = 0 if tp == 0 else tp / (tp + fn)
    f1 = 0 if precision * recall == 0 else 2 * precision * recall / (precision + recall)
    return threshold, f1


def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, patience=5):
    best_threshold = 0.0
    best_val_f1 = 0.0
    epochs_no_improve = 0
    best_model_state = None
    best_epoch = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()  # Set model to training mode
        train_loss = 0.0
        for inputs, targets in tqdm.tqdm(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, targets)
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update the weights

            train_loss += loss.item()

        train_loss /= len(train_loader)
        print("Train loss:", train_loss)
    
        # Validation
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_probs = []  # Store probabilities for ROC-AUC
        print("Validating...")
        with torch.no_grad():
            for inputs, targets in tqdm.tqdm(val_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                # Get predictions and probabilities
                probs = torch.sigmoid(outputs).cpu().numpy()
                labels = targets.cpu().numpy()

                all_labels.extend(labels)
                all_probs.extend(probs.flatten())

        val_loss /= len(val_loader)

        # Find threshold for predictions
        print("Looking for threshold")

        precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
        precisions = precisions[:-1]  # Last value is not real
        recalls = recalls[:-1]  # Last value is not real
        mask = (precisions > 0) & (recalls > 0)

        valid_precisions = precisions[mask]
        valid_recalls = recalls[mask]
        valid_thresholds = thresholds[mask]

        f1_scores = 2 * (valid_precisions * valid_recalls) / (valid_precisions + valid_recalls)
        best_threshold_epoch = valid_thresholds[np.argmax(f1_scores)]

        # Compute metrics with found threshold
        all_preds = (all_probs > best_threshold_epoch).astype(int)
        cm = confusion_matrix(all_labels, all_preds)
        tp = cm[1, 1]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tn = cm[0, 0]

        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0 # Handle division by zero
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        roc_auc = roc_auc_score(all_labels, all_probs)

        print(f"Validation Metrics - Epoch {epoch+1}/{num_epochs}:")
        print(f"Loss      :{val_loss:.4f}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print(f"Confusion Matrix:\n{tp} {fn}\n{fp} {tn}")

        new_row = {
            "model": model_name,
            "year": year,
            "month": month,
            "perc": perc,
            "epoch": latest_epoch + epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "acc": accuracy,
            "prec": precision,
            "rec": recall,
            "f1": f1,
            "auc": roc_auc,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "best_threshold": float(best_threshold_epoch),
            "done": False
        }
        url = "http://localhost:5000/save_results"
        response = requests.post(url, json=new_row)
        assert response.status_code == 200

        torch.save(model.state_dict(), f"./pyg_experiments/model_{model_name}_{year}_{month}_{perc}_{latest_epoch + epoch + 1}.pth")

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_threshold = best_threshold_epoch
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = latest_epoch + epoch + 1
        else:
            epochs_no_improve += 1
            print("Epochs without improving:", epochs_no_improve)
            if epochs_no_improve == patience:
                print(f"Early stopping!!!")
                print(f"Early stopping!!!")
                print(f"Early stopping!!!")
                print("Best epoch:", best_epoch)
                model.load_state_dict(best_model_state)
                break
    
    return best_threshold


def test(model, test_loader, device, criterion, best_threshold):
    model.eval()
    test_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for inputs, targets in tqdm.tqdm(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > best_threshold).astype(int)
            labels = targets.cpu().numpy()

            all_labels.extend(labels)
            all_preds.extend(preds.flatten())
            all_probs.extend(probs.flatten())

    test_loss /= len(test_loader)

    cm = confusion_matrix(all_labels, all_preds)
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tn = cm[0, 0]

    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    try:
      roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        roc_auc = 0.0

    print(f"Test Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"Confusion Matrix:\n{tp} {fn}\n{fp} {tn}")
    print(f"Test Loss: {test_loss:.4f}")


if __name__ == '__main__':
    model_name = "mlp_mb"
    print("model_name", model_name)
    year = 2023
    print("year", year)
    month = 11
    print("month", month)
    perc = 0
    print("perc", perc)
    latest_epoch = 0

    num_epochs = 1000
    patience = 5
    learning_rate = 1e-4
    weight_decay = 1e-5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: '{device}'")

    # x_train = torch.load(f"pyg_experiments/ds/mlp{year}{month}{perc}x_train.pt")
    # x_test = torch.load(f"pyg_experiments/ds/mlp{year}{month}{perc}x_test.pt")
    # y_train = torch.load(f"pyg_experiments/ds/mlp{year}{month}{perc}y_train.pt")
    # y_test = torch.load(f"pyg_experiments/ds/mlp{year}{month}{perc}y_test.pt")

    x_train = torch.load(f"pyg_experiments/ds/mlp{year}{month}{perc}x_train_mb.pt")
    x_test = torch.load(f"pyg_experiments/ds/mlp{year}{month}{perc}x_test_mb.pt")
    y_train = torch.load(f"pyg_experiments/ds/mlp{year}{month}{perc}y_train.pt")
    y_test = torch.load(f"pyg_experiments/ds/mlp{year}{month}{perc}y_test.pt")

    print(f"x_train shape:", x_train.shape)
    print(f"x_test shape:", x_test.shape)
    print(f"y_train shape:", y_train.shape)
    print(f"y_test shape:", y_test.shape)

    input_size = x_train.shape[1]

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_size = int(0.8 * x_train.shape[0])
    val_size = x_train.shape[0] - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=10, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, num_workers=10, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, num_workers=10, pin_memory=True)

    model = MLP(input_size).to(device)
    if latest_epoch > 0:
        model.load_state_dict(torch.load(f"pyg_experiments/model_{model_name}_{year}_{month}_{perc}_{latest_epoch}.pth", weights_only=False))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_threshold = train(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        1000
    )

    torch.save(model.state_dict(), f"./pyg_experiments/trained_models/model_{model_name}_{year}_{month}_{perc}.pth")

    test(
        model,
        test_loader,
        device,
        criterion,
        best_threshold
    )
