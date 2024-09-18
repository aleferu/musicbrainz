#!/usr/bin/env python3


import numpy as np
from sqlalchemy import create_engine, Engine
import pandas as pd
import os
import json
from dotenv import load_dotenv
import logging


def get_engine() -> Engine:
    load_dotenv()

    DB_NAME = os.getenv("DB_NAME")
    DB_HOST = os.getenv("DB_HOST")
    DB_USER = os.getenv("DB_USER")
    DB_PASS = os.getenv("DB_PASS")
    DB_PORT = os.getenv("DB_PORT")
    engine_url = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(engine_url, pool_size=10, max_overflow=0)
    return engine


def get_top_tag_ids() -> set[str]:
    engine = get_engine()

    query = """
        SELECT id
        FROM tag
        WHERE ref_count > 255
    """
    valid_tag_ids = set(pd.read_sql_query(query, engine, dtype=str)["id"])

    engine.dispose()

    return valid_tag_ids


if __name__ == '__main__':
    # logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # First we'll only work with the top tags in MB
    logging.info("Querying for the top tags in MB's DB...")
    top_tags = get_top_tag_ids()
    logging.info(f"Found {len(top_tags)} tags.")

    # Now we'll import our tag CSV
    logging.info("Reading 'tags.csv'...")
    tags_csv = pd.read_csv("tags.csv", dtype=str)
    tags_csv.fillna("", inplace=True)  # We have two tags without name
    logging.info(f"Read {len(tags_csv)} rows.")

    # Import our genre information
    logging.info("Reading 'genres.json'...")
    with open("util/genres.json", "r") as f:
        genres = json.load(f)
    logging.info("Reading 'genres_taxonomy.json'...")
    with open("util/genres_taxonomy.json", "r") as f:
        taxonomy = json.load(f)

    # Error checking
    if genres.keys() != taxonomy.keys():
        logging.error("Taxonomy and genre information is not synchronized.")
        left_in_tax = set(genre for genre in genres.keys() if genre not in taxonomy.keys())
        if len(left_in_tax) > 0:
            logging.error(f"Not found in taxonomy: {", ".join(left_in_tax)}")
        left_in_genres = [tax for tax in taxonomy.keys() if tax not in genres.keys()]
        if len(left_in_genres) > 0:
            logging.error(f"Not found in genres: {", ".join(left_in_genres)}")
        exit()

    logging.info("Everything seems fine!")

    # Name mapping
    logging.info("Creating the name mapping...")
    name_mapping = dict()
    for main, subs in taxonomy.items():
        for sub in subs:
            if sub in name_mapping:
                name_mapping[sub].append(main)
            else:
                name_mapping[sub] = [main]
    logging.info("Done!")

    # tags_clean.csv
    logging.info("Creating the clean version of 'tags.csv'...")
    genre_idx = {
        genre: str(i)
        for i, genre in enumerate(genres.keys())
    }
    temp = pd.DataFrame.from_records(((id, genre) for genre, id in genre_idx.items()), columns=["id", "genre"])
    temp.to_csv("tags_clean.csv", index=False)
    del temp
    logging.info("Done!")

    # Id mapping
    logging.info("Creating the id mapping...")
    homeless = set()
    id_mapping = dict()
    for i in range(len(tags_csv)):
        tag = tags_csv.iloc[i]
        if tag["name"] not in ["us", "dj"] and len(tag["name"]) < 2 or tag["id"] not in top_tags:
            homeless.add((tag["id"], tag["name"]))
            continue
        found = False
        for sub, main_list in name_mapping.items():
            if sub in tag["name"]:
                found = True
                for main in main_list:
                    if tag["id"] in id_mapping:
                        id_mapping[tag["id"]] += f", {genre_idx[main]}"
                    else:
                        id_mapping[tag["id"]] = genre_idx[main]
                break
        if not found:
            homeless.add((tag["id"], tag["name"]))

    logging.info(f"Found {len(id_mapping)} id mappings and {len(homeless)} will be left behind.")

    # Artists
    logging.info("Modifying 'artist_tags.csv'")

    logging.info("  Reading 'artist_tags.csv'...")
    artist_tags = pd.read_csv("artist_tags.csv", dtype=str)

    logging.info("  Modifying data...")
    artist_tags["tags"] = artist_tags["tags"].map(
        lambda tags: ", ".join(set(
            x.strip() for x in (
                (tag_id for tag in tags.split(", ") for tag_id in id_mapping.get(tag.strip(), "").split(", "))
            ) if x != ""
        ))
    )
    artist_tags["tags"] = artist_tags["tags"].replace("", np.nan)
    artist_tags.dropna(subset=["tags"], inplace=True)

    logging.info("  Saving to 'artist_tags_clean.csv'...")
    artist_tags.to_csv("artist_tags_clean.csv", index=False)

    # Releases
    logging.info("Modifying 'releases_no_va_merged_id.csv'")

    logging.info("  Reading 'releases_no_va_merged_id.csv'...")
    releases = pd.read_csv("releases_no_va_merged_id.csv", dtype=str)

    logging.info("  Modifying data...")
    releases["tags"] = releases["tags"].map(
        lambda tags: ", ".join(set(
            x.strip() for x in (
                (tag_id for tag in tags.split(", ") for tag_id in id_mapping.get(tag.strip(), "").split(", "))
            ) if x != ""
        )),
        na_action="ignore"
    )
    releases["tags"] = releases["tags"].replace("", np.nan)

    logging.info("  Saving to 'releases_no_va_merged_id_clean.csv'...")
    releases.to_csv("releases_no_va_merged_id_clean.csv", index=False)

    logging.info("ALL DONE!")
