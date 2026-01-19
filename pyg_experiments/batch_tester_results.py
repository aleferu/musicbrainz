#!/usr/bin/env python3


import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import HeteroConv, GATConv, SAGEConv
import os.path as path
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
import tqdm
import requests
import pickle
import time
import csv


def get_x_map(filepath: str) -> dict:
    with open(filepath, "rb") as in_file:
        loaded_dict = pickle.load(in_file)
    return loaded_dict


def get_artist_map() -> dict:
    print("Loading artist dict...")
    return get_x_map("./pyg_experiments/ds/artist_map.pkl")


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


data_folder = "pyg_experiments/ds/"

# Training parameters
model_name = "nocat_mb"
year = 2019
print("year:", year)
month = 11
print("month:", month)
perc = 0
print("perc:", perc)
train_collab_with_filename = f"collab_withmb_{year}_{month}_{perc}.pt"
model_path = f"pyg_experiments/trained_models/model_{model_name}_{year}_{month}_{perc}.pth"
best_threshold = 0.67

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
model = GNN_NOCAT(metadata=metadata, hidden_channels=64, out_channels=64).to(device)
model.load_state_dict(torch.load(model_path, weights_only=False))

train_collab_with = torch.load(path.join(data_folder, train_collab_with_filename))
train_edges_set = set(map(tuple, train_collab_with.t().tolist()))


def test(model, server_url, device, best_threshold, inv_artist_map):
    model.eval()
    all_edge_data = []  # To store (n_id0, n_id1, prob, pred, label, tp, fp, tn, fn)
    num_batches = 0
    with torch.no_grad():
        for _ in tqdm.tqdm(iter(int, 1)):
            try:
                response = requests.get(f"{server_url}/get_test_batch")
                if response.status_code == 200:
                    num_batches += 1
                    sampled_data = pickle.loads(response.content).to(device)
                    pred_dict = model(sampled_data.x_dict, sampled_data.edge_index_dict)
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

                    # Convert filtered local indices to tensor
                    filtered_edges = torch.tensor(filtered_edges, dtype=torch.long).t().to(device)
                    filtered_labels = torch.tensor(filtered_edge_label).long().to(device)

                    src_emb = pred_dict['artist'][filtered_edges[0]]
                    dst_emb = pred_dict['artist'][filtered_edges[1]]

                    preds = (src_emb * dst_emb).sum(dim=-1)
                    probs = torch.sigmoid(preds)

                    # Collect data for CSV
                    for i in range(filtered_edges.size(1)):
                        src_local_idx = filtered_edges[0][i]
                        dst_local_idx = filtered_edges[1][i]
                        label = filtered_labels[i].item()
                        prob = probs[i].item()
                        pred = prob > best_threshold

                        n_id0 = sampled_data['artist'].n_id[src_local_idx].item()
                        n_id1 = sampled_data['artist'].n_id[dst_local_idx].item()

                        main_id0 = inv_artist_map[n_id0]
                        main_id1 = inv_artist_map[n_id1]

                        tp = 1 if pred == 1 and label == 1 else 0
                        fp = 1 if pred == 1 and label == 0 else 0
                        tn = 1 if pred == 0 and label == 0 else 0
                        fn = 1 if pred == 0 and label == 1 else 0

                        all_edge_data.append([n_id0, n_id1, main_id0, main_id1, prob, pred, label, tp, fp, tn, fn])

                elif response.status_code == 204:
                    break
                else:
                    print(f"Error getting validation batch: {response.status_code}")
                    break
            except requests.exceptions.RequestException as e:
                print(f"Error during validation batch request: {e}")
                time.sleep(5) # Wait before retrying
                continue

    return all_edge_data


def write_to_csv(data, filename="prediction_results.csv"):
    headers = ["n_id0", "n_id1", "main_id0", "main_id1", "prob", "pred", "label", "tp", "fp", "tn", "fn"]
    print(len(data))
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(data)
    print(f"Prediction results written to {filename}")


if __name__ == '__main__':
    artist_map = get_artist_map()
    inv_artist_map = {v: k for k, v in artist_map.items()}

    print(f"Getting test batches... Expected: {test_batches}")
    all_edge_data = test(model, server_url, device, best_threshold, inv_artist_map)
    write_to_csv(all_edge_data, f"prediction_results_{model_name}_{year}_{month}_{perc}.csv")
