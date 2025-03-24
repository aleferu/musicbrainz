#!/usr/bin/env python3


import torch
import pickle
import pandas as pd
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
    df["artist"] = df["artist"].map(artist_map, meta=("artist", str))
    df["tags"] = df["tags"].map(tag_map, meta=("tags", str))
    arr = df.to_dask_array().compute().T
    print(type(arr))
    print(arr)
    print(torch.tensor(arr))


if __name__ == '__main__':
    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    build_artists()
