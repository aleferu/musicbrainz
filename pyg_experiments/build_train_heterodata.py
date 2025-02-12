#!/usr/bin/env python3


import torch
from torch_geometric.data import HeteroData
import os.path as path
import pandas as pd
import logging
import pickle
import numpy as np


def get_full_data() -> HeteroData:
    logging.info("Reading the whole dataset")

    data = HeteroData()

    data["artist"].x = torch.load(path.join(data_folder, "artists.pt"), weights_only=True)
    logging.info(f"  Artist tensor shape: {data["artist"].x.shape}")

    data["track"].x = torch.load(path.join(data_folder, "tracks.pt"), weights_only=True)
    logging.info(f"  Track tensor shape: {data["track"].x.shape}")

    data["tag"].x = torch.load(path.join(data_folder, "tags.pt"), weights_only=True)
    logging.info(f"  Tag tensor shape: {data["tag"].x.shape}")


    data["artist", "collab_with", "artist"].edge_index = torch.load(path.join(data_folder, "collab_with.pt"), weights_only=True)
    data["artist", "collab_with", "artist"].edge_attr = torch.load(path.join(data_folder, "collab_with_attr.pt"), weights_only=True)
    logging.info(f"  collab_with index tensor shape: {data["artist", "collab_with", "artist"].edge_index.shape}")
    logging.info(f"  collab_with attr tensor shape: {data["artist", "collab_with", "artist"].edge_attr.shape}")

    data["artist", "has_tag_artists", "tag"].edge_index = torch.load(path.join(data_folder, "has_tag_artists.pt"), weights_only=True)
    data["track", "has_tag_tracks", "tag"].edge_index = torch.load(path.join(data_folder, "has_tag_tracks.pt"), weights_only=True)
    logging.info(f"  has_tag_artists index tensor shape: {data["artist", "has_tag_artists", "tag"].edge_index.shape}")
    logging.info(f"  has_tag_tracks index tensor shape: {data["track", "has_tag_tracks", "tag"].edge_index.shape}")

    data["artist", "last_fm_match", "artist"].edge_index = torch.load(path.join(data_folder, "last_fm_match.pt"), weights_only=True)
    data["artist", "last_fm_match", "artist"].edge_attr = torch.load(path.join(data_folder, "last_fm_match_attr.pt"), weights_only=True)
    logging.info(f"  last_fm_match index tensor shape: {data["artist", "last_fm_match", "artist"].edge_index.shape}")
    logging.info(f"  last_fm_match attr tensor shape: {data["artist", "last_fm_match", "artist"].edge_attr.shape}")

    data["artist", "linked_to", "artist"].edge_index = torch.load(path.join(data_folder, "linked_to.pt"), weights_only=True)
    data["artist", "linked_to", "artist"].edge_attr = torch.load(path.join(data_folder, "linked_to_attr.pt"), weights_only=True)
    logging.info(f"  linked_to index tensor shape: {data["artist", "linked_to", "artist"].edge_index.shape}")
    logging.info(f"  linked_to attr tensor shape: {data["artist", "linked_to", "artist"].edge_attr.shape}")

    data["artist", "musically_related_to", "artist"].edge_index = torch.load(path.join(data_folder, "musically_related_to.pt"), weights_only=True)
    data["artist", "musically_related_to", "artist"].edge_attr = torch.load(path.join(data_folder, "musically_related_to_attr.pt"), weights_only=True)
    logging.info(f"  musically_related_to index tensor shape: {data["artist", "musically_related_to", "artist"].edge_index.shape}")
    logging.info(f"  musically_related_to attr tensor shape: {data["artist", "musically_related_to", "artist"].edge_attr.shape}")

    data["artist", "personally_related_to", "artist"].edge_index = torch.load(path.join(data_folder, "personally_related_to.pt"), weights_only=True)
    data["artist", "personally_related_to", "artist"].edge_attr = torch.load(path.join(data_folder, "personally_related_to_attr.pt"), weights_only=True)
    logging.info(f"  personally_related_to index tensor shape: {data["artist", "personally_related_to", "artist"].edge_index.shape}")
    logging.info(f"  personally_related_to attr tensor shape: {data["artist", "personally_related_to", "artist"].edge_attr.shape}")

    data["tag", "tags_artists", "artist"].edge_index = torch.load(path.join(data_folder, "tags_artists.pt"), weights_only=True)
    data["tag", "tags_track", "track"].edge_index = torch.load(path.join(data_folder, "tags_tracks.pt"), weights_only=True)
    logging.info(f"  tags_artists index tensor shape: {data["tag", "tags_artists", "artist"].edge_index.shape}")
    logging.info(f"  tags_tracks index tensor shape: {data["tag", "tags_track", "track"].edge_index.shape}")

    data["track", "worked_by", "artist"].edge_index = torch.load(path.join(data_folder, "worked_by.pt"), weights_only=True)
    data["artist", "worked_in", "track"].edge_index = torch.load(path.join(data_folder, "worked_in.pt"), weights_only=True)
    logging.info(f"  worked_by index tensor shape: {data["track", "worked_by", "artist"].edge_index.shape}")
    logging.info(f"  worked_in index tensor shape: {data["artist", "worked_in", "track"].edge_index.shape}")

    if data.validate():
        logging.info("  Full data validation successful!")

    return data


def clean_data(data: HeteroData):
    logging.info(f"Cleaning data per percentile {percentile}")
    edge_types = [
        ("artist", "collab_with", "artist"),
        ("artist", "has_tag_artists", "tag"),
        ("track", "has_tag_tracks", "tag"),
        ("artist", "last_fm_match", "artist"),
        ("artist", "linked_to", "artist"),
        ("artist", "musically_related_to", "artist"),
        ("artist", "personally_related_to", "artist"),
        ("tag", "tags_artists", "artist"),
        ("tag", "tags_track", "track"),
        ("track", "worked_by", "artist"),
        ("artist", "worked_in", "track")
    ]

    # Data
    artist_popularity = data["artist"].x[:, 8]

    # Threshold obtention
    threshold = torch.quantile(artist_popularity, percentile)
    selected_artists = artist_popularity >= threshold
    selected_artist_ids = torch.nonzero(selected_artists).squeeze()

    # Mapping
    old_to_new_artist_idx = torch.zeros(
        data["artist"].x.shape[0],
        dtype=torch.long
    )

    for i, selected_artist_id in enumerate(selected_artist_ids):
        old_to_new_artist_idx[selected_artist_id] = i

    # Subgraph
    for edge_type in edge_types:
        logging.info(f"edge_type: {edge_type}")
        # Filter edge indices
        edge_index = data[edge_type].edge_index
        mask = torch.ones(edge_index.shape[1], dtype=torch.bool)
        if edge_type[0] == "artist":
            mask &= torch.isin(edge_index[0], selected_artist_ids)
        if edge_type[2] == "artist":
            mask &= torch.isin(edge_index[1], selected_artist_ids)

        filtered_edge_index = edge_index[:, mask]

        if edge_type[0] == "artist":  # Reindex source node
            filtered_edge_index[0] = old_to_new_artist_idx[filtered_edge_index[0]]
        if edge_type[2] == "artist":  # Reindex destination node
            filtered_edge_index[1] = old_to_new_artist_idx[filtered_edge_index[1]]

        # Assign filtered edges to subgraph
        data[edge_type].edge_index = filtered_edge_index

        # Handle edge attributes if they exist
        if hasattr(data[edge_type], "edge_attr"):
            try:
                data[edge_type].edge_attr = data[edge_type].edge_attr[mask]
            except IndexError as e:
                logging.info(f"IndexError for {edge_type}: {e}")
        else:
            logging.info(f"No edge_attr for {edge_type}")

    # Nodes filtering
    data["artist"].x = data["artist"].x[selected_artist_ids]
    # data["track"].x = data["track"].x
    # data["tag"].x = data["tag"].x

    # Check the shape of the filtered (or not) nodes and edges
    for edge_type in edge_types:
        logging.info(f"Edge type: {edge_type}, edge_index shape: {data[edge_type].edge_index.shape}")

    # Check the artist features (should only have the selected artists)
    logging.info(f"  Subgraph artist tensor shape: {data["artist"].x.shape}")
    logging.info(f"  Subgraph track tensor shape: {data["track"].x.shape}")
    logging.info(f"  Subgraph tag tensor shape: {data["tag"].x.shape}")

    if data.validate():
        logging.info("  Data validation after percentile cleanup successful!")


def cut_at_date(data: HeteroData) -> HeteroData:
    logging.info(f"Cutting at year {cut_year} and month {cut_month}")

    logging.info("  Reading CSV")
    df = pd.read_csv("data/year_month_track.csv")
    df["track_ids"] = df["track_ids"].apply(eval)

    logging.info("  Setting up some vars")

    # Tracks involved
    mask = (df["year"] < cut_year) | ((df["year"] == cut_year) & (df["month"] < cut_month))
    train_tracks_neo4j = df[mask]["track_ids"].explode().unique().tolist()  # type: ignore

    # Track map
    with open(path.join(data_folder, "track_map.pkl"), "rb") as in_file:
        track_map = pickle.load(in_file)

    # Subgraph definition
    train_tracks_pyg_t = torch.tensor([track_map[track_id] for track_id in train_tracks_neo4j])
    train_artists_pyg = data["artist", "worked_in", "track"].edge_index[0, :][
        torch.isin(data["artist", "worked_in", "track"].edge_index[1, :], train_tracks_pyg_t)
    ]

    train_data = data.subgraph({
        "track": train_tracks_pyg_t,
        "artist": train_artists_pyg
    })

    # Initial edges to consider
    collab_with_edge_index = train_data["artist", "collab_with", "artist"].edge_index[:, ::2]
    worked_in_edge_index = train_data["artist", "worked_in", "track"].edge_index

    # Unique artists that have collabed
    unique_artists = torch.unique(torch.cat((collab_with_edge_index[0, :], collab_with_edge_index[1, :])))

    # Filtering of worked_in
    mask = torch.isin(
        worked_in_edge_index[0, :],
        unique_artists

    )
    filtered_worked_in_edge_index = worked_in_edge_index[:, mask]

    # Find the indices where each artist starts
    artist_id_sorted = filtered_worked_in_edge_index[0, :]

    change_indices = torch.cat((
        torch.tensor([0]),  # Start from index 0
        torch.where(artist_id_sorted[1:] != artist_id_sorted[:-1])[0] + 1
    ))

    # Get artist IDs at those change points
    artists_at_change_points = artist_id_sorted[change_indices]

    # Create dictionary mapping artist → their track indices
    track_ids = filtered_worked_in_edge_index[1, :]
    artist_tracks_dict = {
        artist.item(): track_ids[start:end]
        for artist, start, end in zip(artists_at_change_points, change_indices, torch.cat((change_indices[1:], torch.tensor([track_ids.shape[0]]))))
    }

    # Collect the new collaboration edges
    logging.info("  Building the new collab_with tensors")
    new_collab_with_edge_index = list()
    new_collab_with_edge_attr = list()
    for a0, a1 in zip(collab_with_edge_index[0, :], collab_with_edge_index[1, :]):
        a0_item = a0.item()
        a1_item = a1.item()
        intersection_len = len(np.intersect1d(artist_tracks_dict[a0_item], artist_tracks_dict[a1_item]))
        if intersection_len > 0:
            new_collab_with_edge_index.append((a0_item, a1_item))
            new_collab_with_edge_index.append((a1_item, a0_item))
            new_collab_with_edge_attr.extend([intersection_len, intersection_len])

    train_data["artist", "collab_with", "artist"].edge_index = torch.tensor(new_collab_with_edge_index).t()
    train_data["artist", "collab_with", "artist"].edge_attr = torch.tensor(new_collab_with_edge_attr).t()

    if train_data.validate():
        logging.info("  Data validation after date cut successful!")

    return train_data



def main():
    full_data = get_full_data()

    if percentile > 0:
        clean_data(full_data)

    result = None
    if cut_year is not None:
        result = cut_at_date(full_data)

    logging.info("Saving...")

    result_path = path.join(data_folder, f"full_hd.pt")
    if result is None:
        torch.save(full_data, result_path)
    else:
        torch.save(result, result_path)

    logging.info("Done!")


if __name__ == '__main__':
    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    percentile = 0

    cut_year = None
    cut_month = None

    assert 0 <= percentile <= 1 and (
        (cut_year is None and cut_month is None)
        or
        (cut_year is not None and cut_month is not None)
    )

    data_folder = "pyg_experiments/ds/"
    main()
