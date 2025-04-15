#!/usr/bin/env python3

import torch
from torch_geometric.loader import LinkNeighborLoader
from torch.utils.data import SubsetRandomSampler
import os.path as path
from flask import Flask, jsonify, request
import pickle
import threading
from queue import Queue, Empty

app = Flask(__name__)

# === Config ===
data_folder = "pyg_experiments/ds/"
year = 2019
month = 11
perc = 0.9
train_hd = f"train_hd_{year}_{month}_{perc}.pt"
compt_tree_size = [25, 20]
batch_size = 128
queue_maxsize = 10

# === Load data ===
print("Loading data...")
data = torch.load(path.join(data_folder, train_hd), weights_only=False)
data.contiguous()

edge_indices = torch.arange(data["artist", "collab_with", "artist"].edge_index.shape[1])
num_edges = len(edge_indices)
perm = torch.randperm(num_edges)
split_idx = int(0.8 * num_edges)

train_sampler = SubsetRandomSampler(perm[:split_idx])
val_sampler = SubsetRandomSampler(perm[split_idx:])

# === Create loaders ===
train_loader_base = LinkNeighborLoader(
    data=data,
    num_neighbors=compt_tree_size,
    neg_sampling_ratio=1,
    edge_label_index=("artist", "collab_with", "artist"),
    batch_size=batch_size,
    shuffle=False,
    sampler=train_sampler,
)

val_loader_base = LinkNeighborLoader(
    data=data,
    num_neighbors=compt_tree_size,
    neg_sampling_ratio=1,
    edge_label_index=("artist", "collab_with", "artist"),
    batch_size=batch_size,
    shuffle=False,
    sampler=val_sampler,
)

# === Iterators and Queues ===
train_loader_iter = iter(train_loader_base)
val_loader_iter = iter(val_loader_base)

train_queue = Queue(maxsize=queue_maxsize)
val_queue = Queue(maxsize=queue_maxsize)

# === Prefetching Threads ===
def preload_batches(loader_iter_name: str, queue: Queue):
    def loader_func():
        while True:
            try:
                batch = next(globals()[loader_iter_name])
                pickled = pickle.dumps(batch)
                queue.put(pickled)
            except StopIteration:
                break
    t = threading.Thread(target=loader_func)
    t.daemon = True
    t.start()

# === Start Initial Preloads ===
preload_batches('train_loader_iter', train_queue)
preload_batches('val_loader_iter', val_queue)

# === Routes ===
@app.route('/get_train_batch', methods=['GET'])
def get_train_batch():
    try:
        pickled = train_queue.get(timeout=5)
        return pickled, 200, {'Content-Type': 'application/octet-stream'}
    except Empty:
        return jsonify({"message": "No more training batches"}), 204

@app.route('/get_val_batch', methods=['GET'])
def get_val_batch():
    try:
        pickled = val_queue.get(timeout=5)
        return pickled, 200, {'Content-Type': 'application/octet-stream'}
    except Empty:
        return jsonify({"message": "No more validation batches"}), 204

@app.route('/reset_train_batches', methods=['POST'])
def reset_train_batches():
    global train_loader_iter, train_queue
    train_loader_iter = iter(train_loader_base)
    train_queue = Queue(maxsize=queue_maxsize)
    preload_batches('train_loader_iter', train_queue)
    return jsonify({"message": "Train batches reset"}), 200

@app.route('/reset_val_batches', methods=['POST'])
def reset_val_batches():
    global val_loader_iter, val_queue
    val_loader_iter = iter(val_loader_base)
    val_queue = Queue(maxsize=queue_maxsize)
    preload_batches('val_loader_iter', val_queue)
    return jsonify({"message": "Validation batches reset"}), 200

@app.route('/get_lengths', methods=['GET'])
def get_lengths():
    return jsonify({
        "train": len(train_loader_base),
        "val": len(val_loader_base)
    }), 200

if __name__ == '__main__':
    print("Ready!")
    app.run(debug=False, port=8888)



# #!/usr/bin/env python3

# import torch
# from torch_geometric.loader import LinkNeighborLoader
# from torch.utils.data import SubsetRandomSampler
# import os.path as path
# import numpy as np
# from flask import Flask, request, jsonify
# import pickle

# app = Flask(__name__)

# # Data loading parameters
# data_folder = "pyg_experiments/ds/"
# year = 2023
# month = 11
# perc = 0
# # train_hd = f"train_hdmb_{year}_{month}_{perc}.pt"
# train_hd = f"train_hd_{year}_{month}_{perc}.pt"
# compt_tree_size = [25, 20]
# batch_size = 128

# print("Loading data...")
# data = torch.load(path.join(data_folder, train_hd), weights_only=False)
# data.contiguous()

# edge_indices = torch.arange(data["artist", "collab_with", "artist"].edge_index.shape[1])
# num_edges = len(edge_indices)
# perm = torch.randperm(num_edges)
# split_idx = int(0.8 * num_edges)

# train_sampler = SubsetRandomSampler(perm[:split_idx])
# val_sampler = SubsetRandomSampler(perm[split_idx:])

# print("Creating train_loader...")
# train_loader = LinkNeighborLoader(
#     data=data,
#     num_neighbors=compt_tree_size,
#     neg_sampling_ratio=1,
#     edge_label_index=("artist", "collab_with", "artist"),
#     batch_size=batch_size,
#     shuffle=False,
#     sampler=train_sampler,
# )

# print("Creating val loader...")
# val_loader = LinkNeighborLoader(
#     data=data,
#     num_neighbors=compt_tree_size,
#     neg_sampling_ratio=1,
#     edge_label_index=("artist", "collab_with", "artist"),
#     batch_size=batch_size,
#     shuffle=False,
#     sampler=val_sampler,
# )

# train_iter = iter(train_loader)
# val_iter = iter(val_loader)

# @app.route('/get_lengths', methods=['GET'])
# def get_lengths():
#     return jsonify({
#             "train": len(train_loader),
#             "val": len(val_loader)
#         }), 200

# @app.route('/get_train_batch', methods=['GET'])
# def get_train_batch():
#     global train_iter
#     try:
#         batch = next(train_iter)
#         pickled_batch = pickle.dumps(batch)
#         return pickled_batch, 200, {'Content-Type': 'application/octet-stream'}
#     except StopIteration:
#         train_iter = iter(train_loader)
#         return jsonify({"message": "No more training batches"}), 204

# @app.route('/get_val_batch', methods=['GET'])
# def get_val_batch():
#     global val_iter
#     try:
#         batch = next(val_iter)
#         pickled_batch = pickle.dumps(batch)
#         return pickled_batch, 200, {'Content-Type': 'application/octet-stream'}
#     except StopIteration:
#         val_iter = iter(val_loader)
#         return jsonify({"message": "No more validation batches"}), 204

# if __name__ == '__main__':
#     app.run(debug=False, port=8888) # Set debug to False in production
