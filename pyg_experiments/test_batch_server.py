#!/usr/bin/env python3

import torch
from torch_geometric.loader import LinkNeighborLoader
from torch.utils.data import SubsetRandomSampler
import os.path as path
import numpy as np
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Data loading parameters
data_folder = "pyg_experiments/ds/"
year = 2019
month = 11
perc = 0.75
test_hd = f"full_hdmb_{perc}.pt"
compt_tree_size = [25, 20]
batch_size = 128

print("Loading data...")
data = torch.load(path.join(data_folder, test_hd), weights_only=False)
data.contiguous()

print("Creating test_loader...")
test_loader = LinkNeighborLoader(
    data=data,
    num_neighbors=compt_tree_size,
    neg_sampling_ratio=1,
    edge_label_index=("artist", "collab_with", "artist"),
    batch_size=128,
    shuffle=False,
)

test_iter = iter(test_loader)

@app.route('/get_length', methods=['GET'])
def get_lengths():
    return jsonify({
            "test": len(test_loader),
        }), 200

@app.route('/get_test_batch', methods=['GET'])
def get_val_batch():
    global test_iter
    try:
        batch = next(test_iter)
        pickled_batch = pickle.dumps(batch)
        return pickled_batch, 200, {'Content-Type': 'application/octet-stream'}
    except StopIteration:
        test_iter = iter(test_loader)
        return jsonify({"message": "No more validation batches"}), 204

if __name__ == '__main__':
    app.run(debug=False, port=8889) # Set debug to False in production
