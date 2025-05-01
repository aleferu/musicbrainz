#!/usr/bin/env python3


import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import HeteroConv, GATConv, SAGEConv
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
import tqdm
import copy
import requests
import pickle
import time


# Model definitions (copy from the original script)
class GNN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels):
        super().__init__()
        self.metadata = metadata
        self.out_channels = out_channels

        self.conv1 = HeteroConv({
            ("artist", "collab_with", "artist"): GATConv((artist_channels, artist_channels), hidden_channels, heads=3, concat=False),
            ("artist", "has_tag_artists", "tag"): SAGEConv((artist_channels, tag_channels), hidden_channels, normalize=True, project=True),
            # ("artist", "last_fm_match", "artist"): GATConv((artist_channels, artist_channels), hidden_channels, heads=3, concat=False),
            ("track", "has_tag_tracks", "tag"): SAGEConv((track_channels, tag_channels), hidden_channels, normalize=True, project=True),
            ("artist", "linked_to", "artist"): GATConv((artist_channels, artist_channels), hidden_channels, heads=3, concat=False),
            ("artist", "musically_related_to", "artist"): GATConv((artist_channels, artist_channels), hidden_channels, heads=3, concat=False),
            ("artist", "personally_related_to", "artist"): GATConv((artist_channels, artist_channels), hidden_channels, heads=3, concat=False),
            ("tag", "tags_artists", "artist"): SAGEConv((tag_channels, artist_channels), hidden_channels, normalize=True, project=True),
            ("tag", "tags_tracks", "track"): SAGEConv((tag_channels, track_channels), hidden_channels, normalize=True, project=True),
            ("track", "worked_by", "artist"): SAGEConv((track_channels, artist_channels), hidden_channels, normalize=True, project=True),
            ("artist", "worked_in", "track"): SAGEConv((artist_channels, track_channels), hidden_channels, normalize=True, project=True),
        }, aggr="mean")

        self.conv2 = HeteroConv({
            ("artist", "collab_with", "artist"): GATConv((hidden_channels, hidden_channels), hidden_channels, heads=3, concat=False),
            ("artist", "has_tag_artists", "tag"): SAGEConv((hidden_channels, hidden_channels), hidden_channels, normalize=True, project=True),
            # ("artist", "last_fm_match", "artist"): GATConv((hidden_channels, hidden_channels), hidden_channels, heads=3, concat=False),
            ("track", "has_tag_tracks", "tag"): SAGEConv((hidden_channels, hidden_channels), hidden_channels, normalize=True, project=True),
            ("artist", "linked_to", "artist"): GATConv((hidden_channels, hidden_channels), hidden_channels, heads=3, concat=False),
            ("artist", "musically_related_to", "artist"): GATConv((hidden_channels, hidden_channels), hidden_channels, heads=3, concat=False),
            ("artist", "personally_related_to", "artist"): GATConv((hidden_channels, hidden_channels), hidden_channels, heads=3, concat=False),
            ("tag", "tags_artists", "artist"): SAGEConv((hidden_channels, hidden_channels), hidden_channels, normalize=True, project=True),
            ("tag", "tags_tracks", "track"): SAGEConv((hidden_channels, hidden_channels), hidden_channels, normalize=True, project=True),
            ("track", "worked_by", "artist"): SAGEConv((hidden_channels, hidden_channels), hidden_channels, normalize=True, project=True),
            ("artist", "worked_in", "track"): SAGEConv((hidden_channels, hidden_channels), hidden_channels, normalize=True, project=True),
        }, aggr="mean")

        self.linear1 = Linear(hidden_channels * 2, hidden_channels * 4)
        self.linear2 = Linear(hidden_channels * 4, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict1 = self.conv1(x_dict, edge_index_dict)
        x_dict2 = self.conv2(x_dict1, edge_index_dict)

        x_artist = torch.cat([x_dict1['artist'], x_dict2['artist']], dim=-1)

        x_artist = self.linear1(x_artist)
        x_artist = F.relu(x_artist)
        x_artist = self.linear2(x_artist)

        # Normalize the artist node features
        x_artist = F.normalize(x_artist, p=2, dim=-1)

        # Update the dictionary with the new 'artist' features, leaving other nodes unchanged
        x_dict['artist'] = x_artist

        return x_dict


class GNN_NOCAT(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels):
        super().__init__()
        self.metadata = metadata
        self.out_channels = out_channels

        self.conv1 = HeteroConv({
            ("artist", "collab_with", "artist"): GATConv((artist_channels, artist_channels), hidden_channels, heads=3, concat=False),
            ("artist", "has_tag_artists", "tag"): SAGEConv((artist_channels, tag_channels), hidden_channels, normalize=True, project=True),
            # ("artist", "last_fm_match", "artist"): GATConv((artist_channels, artist_channels), hidden_channels, heads=3, concat=False),
            ("track", "has_tag_tracks", "tag"): SAGEConv((track_channels, tag_channels), hidden_channels, normalize=True, project=True),
            ("artist", "linked_to", "artist"): GATConv((artist_channels, artist_channels), hidden_channels, heads=3, concat=False),
            ("artist", "musically_related_to", "artist"): GATConv((artist_channels, artist_channels), hidden_channels, heads=3, concat=False),
            ("artist", "personally_related_to", "artist"): GATConv((artist_channels, artist_channels), hidden_channels, heads=3, concat=False),
            ("tag", "tags_artists", "artist"): SAGEConv((tag_channels, artist_channels), hidden_channels, normalize=True, project=True),
            ("tag", "tags_tracks", "track"): SAGEConv((tag_channels, track_channels), hidden_channels, normalize=True, project=True),
            ("track", "worked_by", "artist"): SAGEConv((track_channels, artist_channels), hidden_channels, normalize=True, project=True),
            ("artist", "worked_in", "track"): SAGEConv((artist_channels, track_channels), hidden_channels, normalize=True, project=True),
        }, aggr="mean")

        self.conv2 = HeteroConv({
            ("artist", "collab_with", "artist"): GATConv((hidden_channels, hidden_channels), hidden_channels, heads=3, concat=False),
            ("artist", "has_tag_artists", "tag"): SAGEConv((hidden_channels, hidden_channels), hidden_channels, normalize=True, project=True),
            # ("artist", "last_fm_match", "artist"): GATConv((hidden_channels, hidden_channels), hidden_channels, heads=3, concat=False),
            ("track", "has_tag_tracks", "tag"): SAGEConv((hidden_channels, hidden_channels), hidden_channels, normalize=True, project=True),
            ("artist", "linked_to", "artist"): GATConv((hidden_channels, hidden_channels), hidden_channels, heads=3, concat=False),
            ("artist", "musically_related_to", "artist"): GATConv((hidden_channels, hidden_channels), hidden_channels, heads=3, concat=False),
            ("artist", "personally_related_to", "artist"): GATConv((hidden_channels, hidden_channels), hidden_channels, heads=3, concat=False),
            ("tag", "tags_artists", "artist"): SAGEConv((hidden_channels, hidden_channels), hidden_channels, normalize=True, project=True),
            ("tag", "tags_tracks", "track"): SAGEConv((hidden_channels, hidden_channels), hidden_channels, normalize=True, project=True),
            ("track", "worked_by", "artist"): SAGEConv((hidden_channels, hidden_channels), hidden_channels, normalize=True, project=True),
            ("artist", "worked_in", "track"): SAGEConv((hidden_channels, hidden_channels), hidden_channels, normalize=True, project=True),
        }, aggr="mean")

        self.linear1 = Linear(hidden_channels, hidden_channels * 4)
        self.linear2 = Linear(hidden_channels * 4, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict1 = self.conv1(x_dict, edge_index_dict)
        x_dict2 = self.conv2(x_dict1, edge_index_dict)

        # x_artist = torch.cat([x_dict1['artist'], x_dict2['artist']], dim=-1)

        x_artist = self.linear1(x_dict2['artist'])
        x_artist = F.relu(x_artist)
        x_artist = self.linear2(x_artist)

        # Normalize the artist node features
        x_artist = F.normalize(x_artist, p=2, dim=-1)

        # Update the dictionary with the new 'artist' features, leaving other nodes unchanged
        x_dict['artist'] = x_artist

        return x_dict


class GNN_ONECONV(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels):
        super().__init__()
        self.metadata = metadata
        self.out_channels = out_channels

        self.conv1 = HeteroConv({
            ("artist", "collab_with", "artist"): GATConv((artist_channels, artist_channels), hidden_channels, heads=3, concat=False),
            ("artist", "has_tag_artists", "tag"): SAGEConv((artist_channels, tag_channels), hidden_channels, normalize=True, project=True),
            # ("artist", "last_fm_match", "artist"): GATConv((artist_channels, artist_channels), hidden_channels, heads=3, concat=False),
            ("track", "has_tag_tracks", "tag"): SAGEConv((track_channels, tag_channels), hidden_channels, normalize=True, project=True),
            ("artist", "linked_to", "artist"): GATConv((artist_channels, artist_channels), hidden_channels, heads=3, concat=False),
            ("artist", "musically_related_to", "artist"): GATConv((artist_channels, artist_channels), hidden_channels, heads=3, concat=False),
            ("artist", "personally_related_to", "artist"): GATConv((artist_channels, artist_channels), hidden_channels, heads=3, concat=False),
            ("tag", "tags_artists", "artist"): SAGEConv((tag_channels, artist_channels), hidden_channels, normalize=True, project=True),
            ("tag", "tags_tracks", "track"): SAGEConv((tag_channels, track_channels), hidden_channels, normalize=True, project=True),
            ("track", "worked_by", "artist"): SAGEConv((track_channels, artist_channels), hidden_channels, normalize=True, project=True),
            ("artist", "worked_in", "track"): SAGEConv((artist_channels, track_channels), hidden_channels, normalize=True, project=True),
        }, aggr="mean")

        self.linear1 = Linear(hidden_channels, hidden_channels * 4)
        self.linear2 = Linear(hidden_channels * 4, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict1 = self.conv1(x_dict, edge_index_dict)
        # x_dict2 = self.conv2(x_dict1, edge_index_dict)

        # x_artist = torch.cat([x_dict1['artist'], x_dict2['artist']], dim=-1)

        x_artist = self.linear1(x_dict1['artist'])
        x_artist = F.relu(x_artist)
        x_artist = self.linear2(x_artist)

        # Normalize the artist node features
        x_artist = F.normalize(x_artist, p=2, dim=-1)

        # Update the dictionary with the new 'artist' features, leaving other nodes unchanged
        x_dict['artist'] = x_artist

        return x_dict


class GNN_ONECONVONEFF(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels):
        super().__init__()
        self.metadata = metadata
        self.out_channels = out_channels

        self.conv1 = HeteroConv({
            ("artist", "collab_with", "artist"): GATConv((artist_channels, artist_channels), hidden_channels, heads=3, concat=False),
            ("artist", "has_tag_artists", "tag"): SAGEConv((artist_channels, tag_channels), hidden_channels, normalize=True, project=True),
            # ("artist", "last_fm_match", "artist"): GATConv((artist_channels, artist_channels), hidden_channels, heads=3, concat=False),
            ("track", "has_tag_tracks", "tag"): SAGEConv((track_channels, tag_channels), hidden_channels, normalize=True, project=True),
            ("artist", "linked_to", "artist"): GATConv((artist_channels, artist_channels), hidden_channels, heads=3, concat=False),
            ("artist", "musically_related_to", "artist"): GATConv((artist_channels, artist_channels), hidden_channels, heads=3, concat=False),
            ("artist", "personally_related_to", "artist"): GATConv((artist_channels, artist_channels), hidden_channels, heads=3, concat=False),
            ("tag", "tags_artists", "artist"): SAGEConv((tag_channels, artist_channels), hidden_channels, normalize=True, project=True),
            ("tag", "tags_tracks", "track"): SAGEConv((tag_channels, track_channels), hidden_channels, normalize=True, project=True),
            ("track", "worked_by", "artist"): SAGEConv((track_channels, artist_channels), hidden_channels, normalize=True, project=True),
            ("artist", "worked_in", "track"): SAGEConv((artist_channels, track_channels), hidden_channels, normalize=True, project=True),
        }, aggr="mean")

        self.linear = Linear(hidden_channels, hidden_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict1 = self.conv1(x_dict, edge_index_dict)
        # x_dict2 = self.conv2(x_dict1, edge_index_dict)

        # x_artist = torch.cat([x_dict1['artist'], x_dict2['artist']], dim=-1)

        x_artist = self.linear(x_dict1['artist'])

        # Normalize the artist node features
        x_artist = F.normalize(x_artist, p=2, dim=-1)

        # Update the dictionary with the new 'artist' features, leaving other nodes unchanged
        x_dict['artist'] = x_artist

        return x_dict


# Training parameters
model_name = "oneconvoneff_mb"
year = 2019
print("year:", year)
month = 11
print("month:", month)
perc = 0
print("perc:", perc)
latest_epoch = 5
hidden_channels = 64
out_channels = 64
num_epochs = 1000
patience = 5
learning_rate = 1e-4
weight_decay = 1e-5

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")

# Server URL
server_url = "http://localhost:8888"

# Lengths
response = requests.get(f"{server_url}/get_lengths")
assert response.status_code == 200
batches_info = response.json()
train_batches, val_batches = batches_info["train"], batches_info["val"]

# Load a sample batch to get metadata and channel sizes
response = requests.get(f"{server_url}/get_train_batch")
assert response.status_code == 200
sample_train_batch = pickle.loads(response.content)
metadata = sample_train_batch.metadata()
artist_channels = sample_train_batch["artist"].x.size(1)
track_channels = sample_train_batch["track"].x.size(1)
tag_channels = sample_train_batch["tag"].x.size(1)

# Initialize model
# model = GNN_ONECONV(metadata=metadata, hidden_channels=hidden_channels, out_channels=out_channels).to(device)
model = GNN_ONECONVONEFF(metadata=metadata, hidden_channels=hidden_channels, out_channels=out_channels).to(device)
if latest_epoch > 0:
    model.load_state_dict(torch.load(f"pyg_experiments/model_{model_name}_{year}_{month}_{perc}_{latest_epoch}.pth", weights_only=False))
    print("Loaded epoch", latest_epoch)

# Initialize optimizer and loss criterion
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = F.binary_cross_entropy_with_logits


def train_epoch(model, server_url, optimizer, criterion, device):
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    for _ in tqdm.tqdm(iter(int, 1)):
        try:
            response = requests.get(f"{server_url}/get_train_batch")
            if response.status_code == 200:
                sampled_data = pickle.loads(response.content).to(device)
                optimizer.zero_grad()
                pred_dict = model(sampled_data.x_dict, sampled_data.edge_index_dict)
                edge_label_index = sampled_data['artist', 'collab_with', 'artist'].edge_label_index
                edge_label = sampled_data['artist', 'collab_with', 'artist'].edge_label
                src_emb = pred_dict['artist'][edge_label_index[0]]
                dst_emb = pred_dict['artist'][edge_label_index[1]]
                preds = (src_emb * dst_emb).sum(dim=-1)
                loss = criterion(preds, edge_label.float())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
            elif response.status_code == 204:
                break
            else:
                print(f"Error getting train batch: {response.status_code}")
                break
        except requests.exceptions.RequestException as e:
            print(f"Error during training batch request: {e}")
            time.sleep(5) # Wait before retrying
            continue
    return epoch_loss / num_batches if num_batches > 0 else 0


def evaluate_epoch(model, server_url, criterion, device):
    model.eval()
    all_labels = []
    all_probs = []
    val_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for _ in tqdm.tqdm(iter(int, 1)):
            try:
                response = requests.get(f"{server_url}/get_val_batch")
                if response.status_code == 200:
                    sampled_data = pickle.loads(response.content).to(device)
                    pred_dict = model(sampled_data.x_dict, sampled_data.edge_index_dict)
                    edge_label_index = sampled_data['artist', 'collab_with', 'artist'].edge_label_index
                    edge_label = sampled_data['artist', 'collab_with', 'artist'].edge_label
                    src_emb = pred_dict['artist'][edge_label_index[0]]
                    dst_emb = pred_dict['artist'][edge_label_index[1]]
                    preds = (src_emb * dst_emb).sum(dim=-1)
                    loss = criterion(preds, edge_label.float())
                    val_loss += loss.item()
                    probs = torch.sigmoid(preds)
                    all_labels.append(edge_label.cpu())
                    all_probs.append(probs.cpu())
                    num_batches += 1
                elif response.status_code == 204:
                    break
                else:
                    print(f"Error getting validation batch: {response.status_code}")
                    break
            except requests.exceptions.RequestException as e:
                print(f"Error during validation batch request: {e}")
                time.sleep(5) # Wait before retrying
                continue
    all_labels = torch.cat(all_labels) if all_labels else torch.empty(0, dtype=torch.long)
    all_probs = torch.cat(all_probs) if all_probs else torch.empty(0)
    return val_loss / num_batches if num_batches > 0 else 0, all_labels, all_probs

def find_best_threshold(labels, probs):
    best_threshold = 0
    best_f1 = 0
    if labels.numel() > 0:
        for threshold in tqdm.tqdm(np.arange(0.2, 0.91, 0.01)):
            preds_binary = (probs > threshold).long()
            cm = confusion_matrix(labels, preds_binary)
            tp = cm[1, 1]
            fp = cm[0, 1]
            fn = cm[1, 0]
            precision = 0 if tp == 0 else tp / (tp + fp)
            recall = 0 if tp == 0 else tp / (tp + fn)
            f1 = 0 if precision * recall == 0 else 2 * precision * recall / (precision + recall)
            if f1 > best_f1:
                best_threshold = threshold
                best_f1 = f1
    return best_threshold

def calculate_metrics(labels, probs, threshold):
    if labels.numel() == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0
    preds = (probs > threshold).long()
    cm = confusion_matrix(labels, preds)
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tn = cm[0, 0]
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    roc_auc = roc_auc_score(labels, probs)
    return accuracy, precision, recall, f1, roc_auc, tp, fp, fn, tn

if __name__ == '__main__':
    best_val_f1 = 0.0
    best_threshold = 0
    epochs_no_improve = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    best_epoch = 0

    for epoch in range(num_epochs):
        print(f"Getting train batches... Expected: {train_batches}")
        epoch_loss = train_epoch(model, server_url, optimizer, criterion, device)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")

        print(f"Getting val batches... Expected: {val_batches}")
        val_loss, all_labels, all_probs = evaluate_epoch(model, server_url, criterion, device)
        val_losses.append(val_loss)

        assert requests.post(f"{server_url}/reset_train_batches").status_code == 200
        assert requests.post(f"{server_url}/reset_val_batches").status_code == 200

        best_threshold_epoch = find_best_threshold(all_labels, all_probs)
        print(f"Best threshold for this epoch: {best_threshold_epoch}")

        accuracy, precision, recall, f1, roc_auc, tp, fp, fn, tn = calculate_metrics(all_labels, all_probs, best_threshold_epoch)

        print(f"Validation Metrics - Epoch {epoch+1}/{num_epochs}:")
        print(f"Loss:        {val_loss:.4f}")
        print(f"Accuracy:    {accuracy:.4f}")
        print(f"Precision:   {precision:.4f}")
        print(f"Recall:      {recall:.4f}")
        print(f"F1-score:    {f1:.4f}")
        print(f"ROC-AUC:     {roc_auc:.4f}")
        print(f"Confusion Matrix:\n{tp} {fn}\n{fp} {tn}")

        new_row = {
            "model": model_name,
            "year": year,
            "month": month,
            "perc": perc,
            "epoch": latest_epoch + epoch + 1,
            "train_loss": epoch_loss,
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
            "best_threshold": best_threshold_epoch,
            "done": False
        }
        url = "http://localhost:5000/save_results"
        response = requests.post(url, json=new_row)
        assert response.status_code == 200

        torch.save(model.state_dict(), f"pyg_experiments/model_{model_name}_{year}_{month}_{perc}_{latest_epoch + epoch + 1}.pth")

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
                print(f"Best epoch: {best_epoch}")
                model.load_state_dict(best_model_state)
                break

    print("Training complete.")
    print("Best epoch:", best_epoch)
    print("Best validation F1-score:", best_val_f1)
    print("Best threshold:", best_threshold)
    torch.save(model.state_dict(), f"pyg_experiments/trained_models/model_{model_name}_{year}_{month}_{perc}.pth")
