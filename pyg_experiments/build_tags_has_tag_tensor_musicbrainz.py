#!/usr/bin/env python3


import torch
import pickle
import pandas as pd
import dask
from dask import dataframe as dd
import logging
import numpy as np


def get_x_map(filepath: str) -> dict:
    with open(filepath, "rb") as in_file:
        loaded_dict = pickle.load(in_file)
    return loaded_dict


def get_artist_map() -> dict:
    logging.info("Loading artist dict...")
    return get_x_map("./pyg_experiments/ds/artist_map.pkl")


def get_track_map() -> dict:
    logging.info("Loading track dict...")
    return get_x_map("./pyg_experiments/ds/track_map.pkl")


def get_tag_map() -> dict:
    logging.info("Loading tag dict...")
    return get_x_map("./pyg_experiments/ds/tag_map.pkl")


def build_artists():
    df = dd.read_csv("data/artist_tags_clean.csv", dtype=str)
    df["tags"] = df["tags"].str.split(", ")
    df = df.explode("tags")
    artist_map = get_artist_map()
    tag_map = get_tag_map()
    df["artist"] = df["artist"].map_partitions(artist_map, meta=("artist", str))
    df["tags"] = df["tags"].map_partitions(tag_map, meta=("tags", str))
    logging.info("Computing...")
    arr = df.to_dask_array().compute().T
    logging.info("Done! Artist array of shape: %s", str(arr.shape))
    logging.info("Saving artist tensor...")
    arr = torch.tensor(arr, dtype=torch.long)
    torch.save(arr, "pyg_experiments/ds/has_tag_artists_mb.pt")
    arr[[0, 1], :] = arr[[1, 0], :]
    torch.save(arr, "pyg_experiments/ds/tags_artists_mb.pt")


def build_tracks():
    df = dd.read_csv("data/tracks_no_va_merged_id_clean.csv", dtype=str, blocksize="128M")
    df["tags"] = df["tags"].str.split(", ")
    df = df.explode("tags")
    df = df.dropna(subset="tags")
    track_map = get_track_map()
    tag_map = get_tag_map()
    df["id"] = df["id"].map_partitions(track_map, meta=("artist", str))
    df["tags"] = df["tags"].map_partitions(tag_map, meta=("tags", str))
    df = df[["id", "tags"]]
    logging.info("Computing...")
    arr = df.to_dask_array().compute().T
    logging.info("Done! Track array of shape: %s", str(arr.shape))
    logging.info("Saving track tensor...")
    arr = torch.tensor(arr, dtype=torch.long)
    torch.save(arr, "pyg_experiments/ds/has_tag_tracks_mb.pt")
    arr[[0, 1], :] = arr[[1, 0], :]
    torch.save(arr, "pyg_experiments/ds/tags_tracks_mb.pt")


if __name__ == '__main__':
    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # build_artists()
    build_tracks()
