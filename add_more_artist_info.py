#!/usr/bin/env python3


import os
from dotenv import load_dotenv
from typing import Any
import pandas as pd
from sqlalchemy import Engine, create_engine
from neo4j import Driver, GraphDatabase, basic_auth
import itertools
import logging


def execute_query_return(driver: Driver, query: str, params: None | dict[str, Any] = None) -> list[dict[str, Any]]:
    with driver.session() as session:
        result = session.run(query, params)  # type: ignore
        return result.data()


def get_artists(driver: Driver) -> list[dict[str, Any]]:
    query = """
        MATCH (a:Artist) RETURN a.known_ids AS known_ids, a.main_id AS main_id
    """
    artists = execute_query_return(driver, query)
    return artists


def get_artists_info(engine: Engine, artists: list[dict[str, Any]]) -> pd.DataFrame:
    query = f"""
            SELECT id AS id,
            COALESCE(type, -1) AS type,
            COALESCE(gender, -1) AS gender,
            COALESCE(begin_date_year, -1) AS begin_date_year,
            COALESCE(end_date_year, -1) AS end_date_year,
            ended AS ended
        FROM artist
        WHERE id IN ({", ".join(set(id for artist in artists for id in artist["known_ids"]))})
        ;
    """
    return pd.read_sql_query(
        query,
        engine,
        dtype={
            "id": str,
            "type": int,
            "gender": int,
            "begin_date_year": int,
            "end_date_year": int,
            "ended": bool
        }
    )


if __name__ == '__main__':
    load_dotenv()

    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # POSTGRESQL
    POSTGRESQL_NAME = os.getenv("DB_NAME")
    POSTGRESQL_HOST = os.getenv("DB_HOST")
    POSTGRESQL_USER = os.getenv("DB_USER")
    POSTGRESQL_PASS = os.getenv("DB_PASS")
    POSTGRESQL_PORT = os.getenv("DB_PORT")

    assert POSTGRESQL_NAME is not None and \
        POSTGRESQL_HOST is not None and \
        POSTGRESQL_USER is not None and \
        POSTGRESQL_PASS is not None and \
        POSTGRESQL_PORT is not None

    logging.info("Connecting to POSTGRESQL")
    engine_url = f"postgresql://{POSTGRESQL_USER}:{POSTGRESQL_PASS}@{POSTGRESQL_HOST}:{POSTGRESQL_PORT}/{POSTGRESQL_NAME}"
    engine = create_engine(engine_url, pool_size=10, max_overflow=0)

    # NEO4J
    NEO4J_HOST = os.getenv("NEO4J_HOST")
    NEO4J_PORT = os.getenv("NEO4J_PORT")
    NEO4J_USER = os.getenv("NEO4J_USER")
    NEO4J_PASS = os.getenv("NEO4J_PASS")

    assert NEO4J_HOST is not None and \
        NEO4J_PORT is not None and \
        NEO4J_USER is not None and \
        NEO4J_PASS is not None

    logging.info("Connecting to NEO4J")
    driver = GraphDatabase.driver(f"bolt://{NEO4J_HOST}:{NEO4J_PORT}", auth=basic_auth(NEO4J_USER, NEO4J_PASS))

    # Artist info from Neo4j
    logging.info("Querying from Neo4j")
    artists = get_artists(driver)

    # Artist info from PostgreSQL
    logging.info("Querying from PostgreSQL")
    artists_info = get_artists_info(engine, artists)

    # Mask and mapping
    logging.info("Creating mask and mapping")
    mapping = dict(
        (other_id, artist["main_id"])
        for artist in artists for other_id in artist["known_ids"] if other_id != artist["main_id"]
    )
    mask = artists_info["id"].isin(mapping)

    # Changes
    logging.info("Applying mapping")
    artists_info.loc[mask, 'id'] = artists_info.loc[mask, 'id'].map(mapping)

    # One-Hot Encodings
    logging.info("Generating one-hot encodings")
    types = pd.get_dummies(artists_info['type'], prefix='type')
    types.drop(columns="type_-1", inplace=True)
    genders = pd.get_dummies(artists_info['gender'], prefix='gender')
    genders.drop(columns="gender_-1", inplace=True)

    # Grouping everything
    logging.info("Merging everything")
    grouped_artists = pd.concat([artists_info, types, genders], axis=1)
    grouped_artists = grouped_artists.groupby("id").agg({
        **{col: "max" for col in grouped_artists.columns if col.startswith("type_") or col.startswith("gender_")},
        "begin_date_year": lambda values: ",".join(map(str, sorted(value for value in values if value != -1))),
        "end_date_year": lambda values: ",".join(map(str, sorted(value for value in values if value != -1))),
        "ended": "mean"
    })

    # Records
    logging.info("Generating records")
    records = grouped_artists.to_dict("records")  # type: ignore
    query = """
        CALL apoc.periodic.iterate(
            'UNWIND $list_artist_info AS artist_info RETURN artist_info',
            '
                WITH
                    artist_info,
                    SPLIT(artist_info.begin_date_year, ",") AS begin_dates,
                    SPLIT(artist_info.end_date_year, ",") AS end_dates
                MATCH (a:Artist {main_id: artist_info.id})
                SET
                    a.type_1 = artist_info.type_1,
                    a.type_2 = artist_info.type_2,
                    a.type_3 = artist_info.type_3,
                    a.type_4 = artist_info.type_4,
                    a.type_5 = artist_info.type_5,
                    a.type_6 = artist_info.type_6,
                    a.gender_1 = artist_info.gender_1,
                    a.gender_2 = artist_info.gender_2,
                    a.gender_3 = artist_info.gender_3,
                    a.gender_4 = artist_info.gender_4,
                    a.gender_5 = artist_info.gender_5,
                    a.begin_dates = begin_dates,
                    a.end_dates = end_dates,
                    a.ended = artist_info.ended
            ',
            {batchSize: 4000, parallel: true, params: {list_artist_info: $list_artist_info}}
        )
        ;
    """

    # Do the thing
    artist_per_batch = 20_000
    total_artists = len(artists)
    done_count = 0
    for artist_batch in itertools.batched(records, artist_per_batch):
        logging.info(f"Processing batch of size {len(artist_batch)}")
        params = {
            "list_artist_info": artist_batch
        }
        execute_query_return(driver, query, params)
        done_count += len(artist_batch)
        logging.info(f"Processed {done_count} artists out of {total_artists}\n")

    # Cleanup
    driver.close()
    engine.dispose()

    logging.info("DONE!")
