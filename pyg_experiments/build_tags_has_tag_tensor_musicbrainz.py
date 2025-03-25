#!/usr/bin/env python3


import torch
import pickle
import pyspark.sql.functions as F
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import StructType, StructField, StringType
import logging
import numpy as np
import pandas as pd


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
    artist_map = get_artist_map()
    tag_map = get_tag_map()

    artist_broadcast = sc.broadcast(artist_map)
    tag_broadcast = sc.broadcast(tag_map)

    schema = StructType([
        StructField("artist", StringType(), True),
        StructField("tags", StringType(), True),
    ])
    df = (spark.read.format("csv")
        .option("mode", "FAILFAST")
        .option("sep", ",")
        .option("escape", "\"")
        .option("schema", schema)
        .option("lineSep", "\n")
        .option("header", "true")
        .load("data/artist_tags_clean.csv")
    )
    df = df.dropna(subset="tags")

    tags_col = F.col("tags")

    df = df.withColumn("tags", F.split(tags_col, ", "))
    df = df.withColumn("tags", F.explode(tags_col))

    def map_artist_tag(row: Row) -> tuple[int, int]:
        artist_id = artist_broadcast.value.get(row.artist)
        tag_id = tag_broadcast.value.get(row.tags)
        return (artist_id, tag_id)

    rdd = df.rdd.map(map_artist_tag)

    logging.info("Computing...")
    arr = np.array(rdd.collect()).T

    logging.info("Done! Artist array of shape: %s", str(arr.shape))
    logging.info("Saving artist tensor...")
    arr = torch.tensor(arr, dtype=torch.long)
    torch.save(arr, "pyg_experiments/ds/has_tag_artists_mb.pt")
    arr[[0, 1], :] = arr[[1, 0], :]
    torch.save(arr, "pyg_experiments/ds/tags_artists_mb.pt")


def build_tracks():
    track_map = get_track_map()
    tag_map = get_tag_map()

    track_broadcast = sc.broadcast(track_map)
    tag_broadcast = sc.broadcast(tag_map)

    schema = StructType([
        StructField("id", StringType(), True),
        StructField("tags", StringType(), True),
    ])
    df = (spark.read.format("csv")
        .option("mode", "FAILFAST")
        .option("sep", ",")
        .option("escape", "\"")
        .option("schema", schema)
        .option("lineSep", "\n")
        .option("header", "true")
        .load("data/tracks_no_va_merged_id_clean.csv")
        .select("id", "tags")
        .repartition(3)
    )
    df = df.dropna(subset="tags")

    tags_col = F.col("tags")

    df = df.withColumn("tags", F.split(tags_col, ", "))
    df = df.withColumn("tags", F.explode(tags_col))

    def map_track_tag(row: Row) -> tuple[int, int]:
        track_id = track_broadcast.value.get(row.id)
        tag_id = tag_broadcast.value.get(row.tags)
        return (track_id, tag_id)

    mapped_df = df.rdd.map(map_track_tag).toDF(["track", "tag"])

    logging.info("Writing...")
    mapped_df.write.mode("overwrite").parquet("data/track_tag.parquet")

    logging.info("Reading again!")
    pdf = pd.read_parquet("data/track_tag.parquet")
    arr = pdf.to_numpy().T

    logging.info("Done! Track array of shape: %s", str(arr.shape))
    logging.info("Saving track tensor...")
    arr = torch.tensor(arr, dtype=torch.long)
    torch.save(arr, "pyg_experiments/ds/has_tag_tracks_mb.pt")
    arr[[0, 1], :] = arr[[1, 0], :]
    torch.save(arr, "pyg_experiments/ds/tags_tracks_mb.pt")


if __name__ == "__main__":
    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    SPARK_MASTER = "local[*]"

    spark = (SparkSession
        .builder
        .appName("")  # type: ignore
        .config("spark.rdd.compress", "true")
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "2g")
        .config("spark.executor.cores", "2")
        .config("spark.executor.instances", "2")
        .master(SPARK_MASTER)
        .getOrCreate()
    )
    logging.info(f"Driver memory: {spark.conf.get("spark.driver.memory")}")
    logging.info(f"Executor memory: {spark.conf.get("spark.executor.memory")}")
    logging.info(f"Executor cores: {spark.conf.get("spark.executor.cores")}")
    logging.info(f"Executor instances: {spark.conf.get("spark.executor.instances")}")

    sc = spark.sparkContext

    try:
        build_artists()
        build_tracks()
    except Exception as e:
        raise e
    finally:
        spark.stop()
