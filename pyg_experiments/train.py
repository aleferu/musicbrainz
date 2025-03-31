#!/usr/bin/env python3


import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import HeteroConv, GATConv, SAGEConv, Linear
from torch.utils.data import SubsetRandomSampler
import os.path as path
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
import tqdm
import copy
import requests


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


def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, patience=5):
    best_val_f1 = 0.0
    best_threshold = 0
    epochs_no_improve = 0
    best_model_state = None
    train_losses = list()
    val_losses = list()
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        epoch_loss = 0.0
        
        for sampled_data in tqdm.tqdm(train_loader):
            # Move data to device
            sampled_data = sampled_data.to(device)
            
            # Forward pass
            pred_dict = model(sampled_data.x_dict, sampled_data.edge_index_dict)
            
            # Get predictions and labels for the 'collab_with' edge type
            edge_label_index = sampled_data['artist', 'collab_with', 'artist'].edge_label_index
            edge_label = sampled_data['artist', 'collab_with', 'artist'].edge_label

            src_emb = pred_dict['artist'][edge_label_index[0]]  # Source node embeddings
            dst_emb = pred_dict['artist'][edge_label_index[1]]  # Destination node embeddings
            
            # Compute the dot product between source and destination embeddings
            preds = (src_emb * dst_emb).sum(dim=-1)  # Scalar for each edge
            
            # Compute loss
            loss = criterion(preds, edge_label.float())
            epoch_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Average loss for the epoch
        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")

        print("Computing validation metrics")
        
        # Validation metrics
        model.eval()  # Set model to evaluation mode
        all_labels = []
        all_probs = []
        val_loss = 0.0
        
        with torch.no_grad():  # Disable gradient computation for validation
            for sampled_data in tqdm.tqdm(val_loader):
                # Move data to device
                sampled_data = sampled_data.to(device)

                # Forward pass
                pred_dict = model(sampled_data.x_dict, sampled_data.edge_index_dict)

                # Get predictions and labels for the 'collab_with' edge type
                edge_label_index = sampled_data['artist', 'collab_with', 'artist'].edge_label_index
                edge_label = sampled_data['artist', 'collab_with', 'artist'].edge_label

                src_emb = pred_dict['artist'][edge_label_index[0]]  # Source node embeddings
                dst_emb = pred_dict['artist'][edge_label_index[1]]  # Destination node embeddings

                # Compute the dot product between source and destination embeddings
                preds = (src_emb * dst_emb).sum(dim=-1)  # Scalar for each edge

                loss = criterion(preds, edge_label.float())
                val_loss += loss.item()

                probs = torch.sigmoid(preds)  # Convert to probabilities

                # Collect predictions, probabilities, and labels
                all_labels.append(edge_label.cpu())
                all_probs.append(probs.cpu())
        
        # Concatenate all predictions and labels
        all_labels = torch.cat(all_labels)
        all_probs = torch.cat(all_probs)

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Find threshold for predictions
        print("Looking for threshold")
        best_threshold_epoch = 0
        best_f1_epoch = 0
        for threshold in tqdm.tqdm(np.arange(0.2, 0.91, 0.01)):
            preds_binary = (all_probs > threshold).long()
            cm = confusion_matrix(all_labels, preds_binary)
            tp = cm[1, 1]
            fp = cm[0, 1]
            fn = cm[1, 0]
            tn = cm[0, 0]
            precision = 0 if tp == 0 else tp / (tp + fp)
            recall = 0 if tp == 0 else tp / (tp + fn)
            f1 = 0 if precision * recall == 0 else 2 * precision * recall / (precision + recall)
            if f1 > best_f1_epoch:
                best_threshold_epoch = threshold
                best_f1_epoch = f1
        print(f"Best threshold: {best_threshold_epoch}")
        all_preds = (all_probs > best_threshold_epoch).long()
        
        # Compute metrics
        cm = confusion_matrix(all_labels, all_preds)
        tp = cm[1, 1]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tn = cm[0, 0]
        accuracy = (tp + tn) / (tp + fp + fn + tn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        roc_auc = roc_auc_score(all_labels, all_probs)
        
        # Print validation metrics
        print(f"Validation Metrics - Epoch {epoch+1}/{num_epochs}:")
        print(f"Loss:      {val_loss:.4f}")
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
                print(f"Early stopping!!!")
                print(f"Early stopping!!!")
                print("Best epoch:", best_epoch)
                model.load_state_dict(best_model_state)
                break

    return best_threshold


if __name__ == '__main__':
    # data
    data_folder = "pyg_experiments/ds/"
    model_name = "main_nomatch"
    year = 2019
    month = 11
    perc = 0
    latest_epoch = 8
    # train_hd = f"train_hdmb_{year}_{month}_{perc}.pt"
    # train_hd = f"train_hd_{year}_{month}_{perc}.pt"
    train_hd = f"train_hd_nomatch_{year}_{month}_{perc}.pt"
    print("model_name:", model_name)
    print("year:", year)
    print("month:", month)
    print("perc:", perc)
    print("latest_epoch:", latest_epoch)
    print("train_hd:", train_hd)

    # heterodata
    data = torch.load(path.join(data_folder, train_hd))
    data.validate()

    artist_channels = data["artist"].x.size(1)
    track_channels = data["track"].x.size(1)
    tag_channels = data["tag"].x.size(1)

    print(f"Artist channels: {artist_channels}")
    print(f"Track channels: {track_channels}")
    print(f"Tag channels: {tag_channels}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data.contiguous()

    print(f"Device: '{device}'")

    # loaders
    compt_tree_size = [25, 20]

    edge_indices = torch.arange(data["artist", "collab_with", "artist"].edge_index.shape[1])

    # Shuffle and split
    num_edges = len(edge_indices)
    perm = torch.randperm(num_edges)
    split_idx = int(0.8 * num_edges)

    train_sampler = SubsetRandomSampler(perm[:split_idx])  # type: ignore
    val_sampler = SubsetRandomSampler(perm[split_idx:])  # type: ignore

    print("Creating train_loader...")
    train_loader = LinkNeighborLoader(
        data=data,
        num_neighbors=compt_tree_size,
        neg_sampling_ratio=1,
        edge_label_index=("artist", "collab_with", "artist"),
        batch_size=128,
        shuffle=False,
        num_workers=10,
        pin_memory=True,
        sampler=train_sampler,
    )

    print("Creating val loader...")
    val_loader = LinkNeighborLoader(
        data=data,
        num_neighbors=compt_tree_size,
        neg_sampling_ratio=1,
        edge_label_index=("artist", "collab_with", "artist"),
        batch_size=128,
        shuffle=False,
        num_workers=10,
        pin_memory=True,
        sampler=val_sampler,
    )

    print("Number of train batches:", len(train_loader))
    print("Number of validation batches:", len(val_loader))

    model = GNN(metadata=data.metadata(), hidden_channels=64, out_channels=64).to(device)

    if latest_epoch > 0:
        model.load_state_dict(torch.load(f"pyg_experiments/model_{model_name}_{year}_{month}_{perc}_{latest_epoch}.pth"))
        print("Loaded epoch", latest_epoch)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)  # type: ignore

    best_threshold = train(
        model,
        train_loader,
        val_loader,
        optimizer,
        F.binary_cross_entropy_with_logits,
        device,
        100
    )

    print("model_name:", model_name)
    print("year:", year)
    print("month:", month)
    print("perc:", perc)
    print("latest_epoch:", latest_epoch)
    print("train_hd:", train_hd)
    print("BEST THRESHOLD:", best_threshold)

    torch.save(model.state_dict(), f"pyg_experiments/trained_models/model_{model_name}_{year}_{month}_{perc}.pth")
