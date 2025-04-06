#!/usr/bin/env python3


from neo4j import GraphDatabase, basic_auth, Driver
from dotenv import load_dotenv
import os
import logging
from itertools import combinations_with_replacement
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def main(driver: Driver, tags: list[str], filename: str) -> None:
    combined_tags = [tag for tag in tags if tag != "symphonic" and tag != "classical and ost"]
    if "symphonic" in tags or "classical and ost" in tags:
        combined_tags.append("classical/symphonic")

    heatmap_data = pd.DataFrame(
        index=combined_tags,  # type: ignore
        columns=combined_tags,  # type: ignore
        dtype=int
    ).fillna(0).astype(int)

    # Other
    with driver.session() as session:
        for t0, t1 in combinations_with_replacement(tags, 2):
            logging.info("Querying for '%s' and '%s'...", t0, t1)
            query = f"""
                MATCH (n:Artist {{main_tag: "{t0}"}})-[r:COLLAB_WITH]->(m:Artist {{main_tag: "{t1}"}})
                WHERE n < m
                WITH COUNT(r) AS c
                RETURN toInteger(c) AS c;
            """
            if t0 == "a" * 10 or t1 == "a" * 10:
                count = 0
            else:
                result = session.run(query)  # type: ignore
                count = result.data()[0]["c"]

            logging.info("Got: %d", count)

            if t0 == "symphonic" or t0 == "classical and ost":
                t0_heatmap = "classical/symphonic"
            else:
                t0_heatmap = t0

            if t1 == "symphonic" or t1 == "classical and ost":
                t1_heatmap = "classical/symphonic"
            else:
                t1_heatmap = t1

            heatmap_data.loc[t0_heatmap, t1_heatmap] += count
            heatmap_data.loc[t1_heatmap, t0_heatmap] += count

    # Normalization by row
    row_sums = heatmap_data.sum(axis=1)
    normalized_heatmap_data = heatmap_data.div(row_sums, axis=0)

    # Drawing
    logging.info("Generating png...")

    plt.figure(figsize=(12, 12))
    sns.heatmap(
        normalized_heatmap_data,
        annot=True,
        cmap="coolwarm",
        xticklabels=True,
        yticklabels=True,
        cbar=True,
        square=True,
        fmt=".3f",
        linewidths=.5,
        linecolor='black',
        cbar_kws={"shrink": .8}
    )
    plt.tight_layout()
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.savefig(f"img/{filename}.png")

    # mask = np.zeros((len(combined_tags), len(combined_tags)), dtype=bool)
    # mask[np.eye(mask.shape[0], dtype=bool)] = True

    # plt.figure(figsize=(12, 12))
    # sns.heatmap(
    #     heatmap_data,
    #     annot=True,
    #     cmap="coolwarm",
    #     xticklabels=True,
    #     yticklabels=True,
    #     mask=mask,
    #     cbar=True,
    #     fmt="d",
    #     linewidths=.5,
    #     linecolor='black',
    #     cbar_kws={"shrink": .8}
    # )
    # plt.tight_layout()
    # plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    # plt.savefig(f"img/{filename}_no_diag.png")

    # mask = np.ones((len(combined_tags), len(combined_tags)), dtype=bool)
    # for i in range(len(combined_tags)):
    #     mask[i, :i + 1] = False

    # plt.figure(figsize=(12, 12))
    # sns.heatmap(
    #     heatmap_data,
    #     annot=True,
    #     cmap="coolwarm",
    #     xticklabels=True,
    #     yticklabels=True,
    #     mask=mask,
    #     cbar=True,
    #     fmt="d",
    #     linewidths=.5,
    #     linecolor='black',
    #     cbar_kws={"shrink": .8}
    # )
    # plt.tight_layout()
    # plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    # plt.savefig(f"img/{filename}_triang.png")

    # mask = np.ones((len(combined_tags), len(combined_tags)), dtype=bool)
    # for i in range(len(combined_tags)):
    #     mask[i, :i] = False

    # plt.figure(figsize=(12, 12))
    # sns.heatmap(
    #     heatmap_data,
    #     annot=True,
    #     cmap="coolwarm",
    #     xticklabels=True,
    #     yticklabels=True,
    #     mask=mask,
    #     cbar=True,
    #     fmt="d",
    #     linewidths=.5,
    #     linecolor='black',
    #     cbar_kws={"shrink": .8}
    # )
    # plt.tight_layout()
    # plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    # plt.savefig(f"img/{filename}_triang_no_diag.png")


if __name__ == '__main__':
    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # .env read
    load_dotenv()
    DB_HOST = os.getenv("NEO4J_HOST")
    DB_PORT = os.getenv("NEO4J_PORT")
    DB_USER = os.getenv("NEO4J_USER")
    DB_PASS = os.getenv("NEO4J_PASS")

    # .env validation
    assert DB_HOST is not None and \
        DB_PORT is not None and \
        DB_USER is not None and \
        DB_PASS is not None, \
        "INVALID .env"

    # db connection
    driver = GraphDatabase.driver(f"bolt://{DB_HOST}:{DB_PORT}", auth=basic_auth(DB_USER, DB_PASS))

    test = ["rock", "pop"] + ["a" * 10] * 10

    good_tags = [
        "rock",
        "folk/traditional",
        "pop",
        "electronic",
        "rythm and blues",
        "indie",
        "hip-hop",
        "latin",
        "classical and ost",
        "symphonic",
        "latin_countries"
    ]
    bad_tags = [
        "country",
        "happy",
        "romantic",
        "sad",
        "avant-garde",
        "religion",
        "asian_countries",
        "english_countries",
        "sci-fi",
        "acoustic",
        "underground",
        "drum and bass"
    ]

    # main(driver, test, "test")

    main(driver, good_tags, "good_coll_tags")

    # main(driver, bad_tags, "bad_coll_tags")

    # main(driver, good_tags + bad_tags, "all_coll_tags")

    driver.close()

    logging.info("DONE!")
