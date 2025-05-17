#!/usr/bin/env python3


import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import HeteroConv, GATConv, SAGEConv, GATv2Conv
import os.path as path
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
import tqdm
import requests
import pickle
import time


class GNN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels):
        super().__init__()
        self.metadata = metadata
        self.out_channels = out_channels

        self.conv1 = HeteroConv({
            ("artist", "collab_with", "artist"): GATConv((artist_channels, artist_channels), hidden_channels, heads=3, concat=False),
            ("artist", "has_tag_artists", "tag"): SAGEConv((artist_channels, tag_channels), hidden_channels, normalize=True, project=True),
            ("artist", "last_fm_match", "artist"): GATv2Conv((artist_channels, artist_channels), hidden_channels, heads=3, concat=False, edge_dim=1),
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
            ("artist", "last_fm_match", "artist"): GATv2Conv((hidden_channels, hidden_channels), hidden_channels, heads=3, concat=False, edge_dim=1),
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

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict1 = self.conv1(x_dict, edge_index_dict, edge_attr_dict)
        x_dict2 = self.conv2(x_dict1, edge_index_dict, edge_attr_dict)

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
            ("artist", "last_fm_match", "artist"): GATv2Conv((artist_channels, artist_channels), hidden_channels, heads=3, concat=False, edge_dim=1),
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
            ("artist", "last_fm_match", "artist"): GATv2Conv((hidden_channels, hidden_channels), hidden_channels, heads=3, concat=False, edge_dim=1),
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

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict1 = self.conv1(x_dict, edge_index_dict, edge_attr_dict)
        x_dict2 = self.conv2(x_dict1, edge_index_dict, edge_attr_dict)

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
            ("artist", "last_fm_match", "artist"): GATv2Conv((artist_channels, artist_channels), hidden_channels, heads=3, concat=False, edge_dim=1),
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

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict1 = self.conv1(x_dict, edge_index_dict, edge_attr_dict)
        # x_dict2 = self.conv2(x_dict1, edge_index_dict, edge_attr_dict)

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
            ("artist", "last_fm_match", "artist"): GATv2Conv((artist_channels, artist_channels), hidden_channels, heads=3, concat=False, edge_dim=1),
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

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict1 = self.conv1(x_dict, edge_index_dict, edge_attr_dict)
        # x_dict2 = self.conv2(x_dict1, edge_index_dict, edge_attr_dict)

        # x_artist = torch.cat([x_dict1['artist'], x_dict2['artist']], dim=-1)

        x_artist = self.linear(x_dict1['artist'])

        # Normalize the artist node features
        x_artist = F.normalize(x_artist, p=2, dim=-1)

        # Update the dictionary with the new 'artist' features, leaving other nodes unchanged
        x_dict['artist'] = x_artist

        return x_dict


data_folder = "pyg_experiments/ds/"

# Training parameters
model_name = "main_lfm"
year = 2021
print("year:", year)
month = 11
print("month:", month)
perc = 0.5
print("perc:", perc)
hidden_channels = 64
out_channels = 64
test_hd = f"full_hd_{perc}.pt"
train_collab_with_filename = f"collab_with_{year}_{month}_{perc}.pt"
model_path = f"pyg_experiments/trained_models/model_{model_name}_{year}_{month}_{perc}.pth"
best_threshold = 0.72

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")

# Server URL
server_url = "http://localhost:8889"

# Lengths
response = requests.get(f"{server_url}/get_length")
assert response.status_code == 200
batches_info = response.json()
test_batches = batches_info["test"]

# Load a sample batch to get metadata and channel sizes
response = requests.get(f"{server_url}/get_test_batch")
assert response.status_code == 200
sample_train_batch = pickle.loads(response.content)
metadata = sample_train_batch.metadata()
artist_channels = sample_train_batch["artist"].x.size(1)
track_channels = sample_train_batch["track"].x.size(1)
tag_channels = sample_train_batch["tag"].x.size(1)

# Initialize model
# model = GNN_ONECONV(metadata=metadata, hidden_channels=64, out_channels=64).to(device)
model = GNN(metadata=metadata, hidden_channels=hidden_channels, out_channels=out_channels).to(device)
model.load_state_dict(torch.load(model_path, weights_only=False))

# Initialize optimizer and loss criterion
criterion = F.binary_cross_entropy_with_logits

train_collab_with = torch.load(path.join(data_folder, train_collab_with_filename))
train_edges_set = set(map(tuple, train_collab_with.t().tolist()))


def test(model, server_url, criterion, device):
    model.eval()
    all_labels = []
    all_probs = []
    test_loss = 0.0
    num_batches = 0
    valid_batches = 0
    with torch.no_grad():
        for _ in tqdm.tqdm(iter(int, 1)):
            try:
                response = requests.get(f"{server_url}/get_test_batch")
                if response.status_code == 200:
                    num_batches += 1
                    sampled_data = pickle.loads(response.content).to(device)

                    # Create a new edge_attr_dict containing only 'last_fm_match' attributes
                    last_fm_match_edge_attr = {}
                    if ("artist", "last_fm_match", "artist") in sampled_data.edge_attr_dict:
                        last_fm_match_edge_attr[("artist", "last_fm_match", "artist")] = sampled_data.edge_attr_dict[("artist", "last_fm_match", "artist")]

                    # Forward pass
                    pred_dict = model(sampled_data.x_dict, sampled_data.edge_index_dict, last_fm_match_edge_attr)

                    edge_label_index = sampled_data['artist', 'collab_with', 'artist'].edge_label_index
                    edge_label = sampled_data['artist', 'collab_with', 'artist'].edge_label

                    # Filter edge list
                    filtered_edges = []
                    filtered_edge_label = []
                    positive_count = 0
                    for src, dst, label in zip(edge_label_index[0, :], edge_label_index[1, :], edge_label):
                        lookup_edge = (
                            sampled_data['artist'].n_id[src].item(),
                            sampled_data['artist'].n_id[dst].item()
                        )
                        if lookup_edge in train_edges_set:
                            continue
                        label_item = label.item()

                        # Balancing
                        if np.isclose(label_item, 1):
                            filtered_edge_label.append(label_item)
                            filtered_edges.append([src.item(), dst.item()])
                            positive_count += 1

                        elif positive_count > 0:
                            filtered_edge_label.append(label_item)
                            filtered_edges.append([src.item(), dst.item()])
                            positive_count -= 1
                        else:
                            break

                    if len(filtered_edges) == 0:
                        continue  # Skip if no valid edges left
                    
                    valid_batches += 1

                    # Normal evaluation with the rests
                    filtered_edges = torch.tensor(filtered_edges, dtype=torch.long).t().to(device)
                    filtered_labels = torch.tensor(filtered_edge_label).long().to(device)

                    src_emb = pred_dict['artist'][filtered_edges[0]]  # Source node embeddings
                    dst_emb = pred_dict['artist'][filtered_edges[1]]  # Destination node embeddings

                    # Compute the dot product between source and destination embeddings
                    preds = (src_emb * dst_emb).sum(dim=-1)  # Scalar for each edge
                    probs = torch.sigmoid(preds)  # Convert logits to probabilities

                    loss = criterion(preds, filtered_labels.float())
                    test_loss += loss.item()
                    probs = torch.sigmoid(preds)
                    all_labels.append(filtered_labels.cpu())
                    all_probs.append(probs.cpu())
                elif response.status_code == 204:
                    break
                else:
                    print(f"Error getting validation batch: {response.status_code}")
                    break
            except requests.exceptions.RequestException as e:
                print(f"Error during validation batch request: {e}")
                time.sleep(5) # Wait before retrying
                continue
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

    return test_loss / valid_batches if valid_batches > 0 else 0, all_labels, all_probs


def calculate_metrics(labels, probs, threshold):
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
    print(f"Getting test batches... Expected: {test_batches}")
    test_loss, all_labels, all_probs = test(model, server_url, criterion, device)

    accuracy, precision, recall, f1, roc_auc, tp, fp, fn, tn = calculate_metrics(all_labels, all_probs, best_threshold)

    print(f"Test Metrics:")
    print(f"Loss:        {test_loss:.4f}")
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1-score:    {f1:.4f}")
    print(f"ROC-AUC:     {roc_auc:.4f}")
    print(f"Confusion Matrix:\n{tp} {fn}\n{fp} {tn}")
