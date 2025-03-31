#!/usr/bin/env python3


import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import HeteroConv, GATConv, SAGEConv, Linear
import os.path as path
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import tqdm
import logging


class GNN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels):
        super().__init__()
        self.metadata = metadata
        self.out_channels = out_channels

        self.conv1 = HeteroConv({
            ("artist", "collab_with", "artist"): GATConv((-1, -1), hidden_channels),
            ("artist", "has_tag_artists", "tag"): SAGEConv((-1, -1), hidden_channels),
            ("artist", "last_fm_match", "artist"): GATConv((-1, -1), hidden_channels),
            ("track", "has_tag_tracks", "tag"): SAGEConv((-1, -1), hidden_channels),
            ("artist", "linked_to", "artist"): GATConv((-1, -1), hidden_channels),
            ("artist", "musically_related_to", "artist"): GATConv((-1, -1), hidden_channels),
            ("artist", "personally_related_to", "artist"): GATConv((-1, -1), hidden_channels),
            ("tag", "tags_artists", "artist"): SAGEConv((-1, -1), hidden_channels),
            ("tag", "tags_tracks", "track"): SAGEConv((-1, -1), hidden_channels),
            ("track", "worked_by", "artist"): SAGEConv((-1, -1), hidden_channels),
            ("artist", "worked_in", "track"): SAGEConv((-1, -1), hidden_channels),
        }, aggr="mean")

        self.conv2 = HeteroConv({
            ("artist", "collab_with", "artist"): GATConv((-1, -1), hidden_channels),
            ("artist", "has_tag_artists", "tag"): SAGEConv((-1, -1), hidden_channels),
            ("artist", "last_fm_match", "artist"): GATConv((-1, -1), hidden_channels),
            ("track", "has_tag_tracks", "tag"): SAGEConv((-1, -1), hidden_channels),
            ("artist", "linked_to", "artist"): GATConv((-1, -1), hidden_channels),
            ("artist", "musically_related_to", "artist"): GATConv((-1, -1), hidden_channels),
            ("artist", "personally_related_to", "artist"): GATConv((-1, -1), hidden_channels),
            ("tag", "tags_artists", "artist"): SAGEConv((-1, -1), hidden_channels),
            ("tag", "tags_tracks", "track"): SAGEConv((-1, -1), hidden_channels),
            ("track", "worked_by", "artist"): SAGEConv((-1, -1), hidden_channels),
            ("artist", "worked_in", "track"): SAGEConv((-1, -1), hidden_channels),
        }, aggr="mean")

        self.linear1 = Linear(hidden_channels * 2, hidden_channels * 4)
        self.linear2 = Linear(hidden_channels * 4, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict1 = self.conv1(x_dict, edge_index_dict)
        x_dict2 = self.conv2(x_dict1, edge_index_dict)

        x_artist = torch.cat([x_dict1['artist'], x_dict2['artist']], dim=-1)

        x_artist = self.linear1(x_artist)
        x_artist = self.linear2(x_artist)

        # Normalize the artist node features
        x_artist = F.normalize(x_artist, p=2, dim=-1)

        # Update the dictionary with the new 'artist' features, leaving other nodes unchanged
        x_dict['artist'] = x_artist

        return x_dict


def train(model, train_loader, optimizer, criterion, device, num_epochs, val_epochs):
    for epoch in range(num_epochs):
        logging.info("  Epoch %d", epoch)
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

        # Validation?
        if (epoch + 1) % val_epochs != 0:
            continue

        
        # Validation metrics
        model.eval()  # Set model to evaluation mode
        all_labels = []
        all_probs = []
        
        with torch.no_grad():  # Disable gradient computation for validation
            logging.info("  Training evaluation")
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

                probs = torch.sigmoid(preds)  # Convert to probabilities
                
                # Collect predictions, probabilities, and labels
                all_labels.append(edge_label.cpu())
                all_probs.append(probs.cpu())
        
        # Concatenate all predictions and labels
        all_labels = torch.cat(all_labels)
        all_probs = torch.cat(all_probs)

        # Find threshold for predictions
        logging.info("  Threshold...")
        best_threshold = 0
        best_f1 = 0
        for threshold in tqdm.tqdm(np.arange(0.2, 0.81, 0.01)):
            preds_binary = (all_probs > threshold).long()
            cm = confusion_matrix(all_labels, preds_binary)
            tp = cm[1, 1]
            fp = cm[0, 1]
            fn = cm[1, 0]
            tn = cm[0, 0]
            precision = 0 if tp == 0 else tp / (tp + fp)
            recall = 0 if tp == 0 else tp / (tp + fn)
            f1 = 0 if precision * recall == 0 else 2 * precision * recall / (precision + recall)
            if f1 > best_f1:
                best_threshold = threshold
                best_f1 = f1

    return epoch_loss, best_threshold  # type: ignore


def test_model(model, test_loader, device, threshold, train_edges_set):
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():  # Disable gradient computation
        logging.info("  Testing")
        for sampled_data in tqdm.tqdm(test_loader):
            # Move data to the device
            sampled_data = sampled_data.to(device)

            # Forward pass
            pred_dict = model(sampled_data.x_dict, sampled_data.edge_index_dict)

            # Get predictions and labels for the 'collab_with' edge type
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
                filtered_edges.append([src.item(), dst.item()])
                label_item = label.item()
                filtered_edge_label.append(label_item)

                # Balancing
                if np.isclose(label_item, 1):
                    positive_count += 1
                else:
                    positive_count -= 1
                    if positive_count == 0:
                        break

            if len(filtered_edges) == 0:
                continue  # Skip if no valid edges left

            # Normal evaluation with the rests
            filtered_edges = torch.tensor(filtered_edges, dtype=torch.long).t().to(device)
            filtered_labels = torch.tensor(filtered_edge_label).to(device)

            src_emb = pred_dict['artist'][filtered_edges[0]]  # Source node embeddings
            dst_emb = pred_dict['artist'][filtered_edges[1]]  # Destination node embeddings

            # Compute the dot product between source and destination embeddings
            preds = (src_emb * dst_emb).sum(dim=-1)  # Scalar for each edge
            probs = torch.sigmoid(preds)  # Convert logits to probabilities
            preds_binary = (probs > threshold).long()  # Convert probabilities to binary predictions

            # Collect predictions and labels
            all_preds.append(preds_binary.cpu())
            all_labels.append(filtered_labels.cpu())
            all_probs.append(probs.cpu())

    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

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

    return accuracy, precision, recall, f1, roc_auc


def main():
    model = None
    optimizer = None
    results = list()
    for h in [64, 128, 256]:
        for o in [64, 128, 256]:
            if model is not None or optimizer is not None:
                del model
                del optimizer

            logging.info("Model with h = %d, o = %d", h, o)

            model = GNN(metadata=train_data.metadata(), hidden_channels=h, out_channels=o).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # type: ignore

            train_loss, best_threshold = train(
                model,
                train_loader,
                optimizer,
                F.binary_cross_entropy_with_logits,
                device,
                3,
                3
            )

            accuracy, precision, recall, f1, roc_auc = test_model(
                model,
                test_loader,
                device,
                best_threshold,
                train_edges_set
            )

            logging.info(f"  train_loss: {train_loss}")
            logging.info(f"  accuracy: {accuracy}")
            logging.info(f"  precision: {precision}")
            logging.info(f"  recall: {recall}")
            logging.info(f"  f1: {f1}")
            logging.info(f"  roc_auc: {roc_auc}")

            results.append([
                h,
                o,
                train_loss,
                accuracy,
                precision,
                recall,
                f1,
                roc_auc
            ])

    logging.info("Writing file!")
    with open("ho_results.csv", "w") as out_f:
        out_f.write("h,o,loss,acc,pre,rec,f1,auc\n")
        for result in results:
            out_f.write(f"{result.join(",")}\n")


if __name__ == '__main__':
    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    compt_tree_size = [25, 20]
    logging.info(f"Compt_tree_size: {compt_tree_size}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: '{device}'")

    data_folder = "pyg_experiments/ds/"
    train_hd = "train_hd_2020_3_0.9.pt"
    test_hd = "full_hd_0.9.pt"
    train_collab_with_filename = "collab_with_2020_3_0.9.pt"

    logging.info("Loading training data...")
    train_data = torch.load(path.join(data_folder, train_hd))
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=compt_tree_size,
        neg_sampling_ratio=1,
        edge_label_index=("artist", "collab_with", "artist"),
        batch_size=128,
        shuffle=True,
        num_workers=10,
        pin_memory=True,
    )

    logging.info("Loading test data...")
    test_data = torch.load(path.join(data_folder, test_hd))
    test_loader = LinkNeighborLoader(
        data=test_data,
        num_neighbors=compt_tree_size,
        neg_sampling_ratio=1,
        edge_label_index=("artist", "collab_with", "artist"),
        batch_size=128,
        num_workers=10,
        pin_memory=True,
    )

    train_collab_with = torch.load(path.join(data_folder, train_collab_with_filename))
    train_edges_set = set(map(tuple, train_collab_with.t().tolist()))

    main()
