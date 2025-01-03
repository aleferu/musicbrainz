#!/usr/bin/env python3


from neo4j import GraphDatabase, basic_auth, Driver
from neo4j import AsyncDriver, AsyncGraphDatabase
from dotenv import load_dotenv
import pandas as pd
from sklearn.decomposition import PCA
import os
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
import logging
import asyncio
from itertools import batched


def artist_popularity(driver: Driver) -> None:
    # Pandas
    logging.info("Querying and building the main dataframe...")
    with driver.session() as session:
        query = """
            MATCH (n:Artist)
            RETURN
                n.main_id AS main_id,
                n.listeners AS listeners,
                n.playcount AS playcount,
                n.in_last_fm AS in_last_fm
            ;
        """
        q_result = session.run(query)
        df = pd.DataFrame(q_result.data())
    logging.info(f"Found {len(df)} artists")

    # df with artists that have info
    logging.info(f"Found {df["in_last_fm"].sum()} artists with popularity information.")

    # Corr
    r, p_value = pearsonr(df.listeners, df.playcount)
    logging.info(f"Correlation between listeners and playcount: {r}.")
    logging.info(f"Correlation p_value: {p_value}. Significative? {p_value < 0.05}")  # type: ignore

    # Scaling
    mmscaler = MinMaxScaler()
    df["listeners_scaled"] = mmscaler.fit_transform(df[["listeners"]])
    df["playcount_scaled"] = mmscaler.fit_transform(df[["playcount"]])

    # PCA
    pca_fitter = PCA(1)
    pca = pca_fitter.fit_transform(df[["listeners_scaled", "playcount_scaled"]])
    df["popularity"] = pca
    pca_scaled = mmscaler.fit_transform(pca)
    df["popularity_scaled"] = pca_scaled

    var_explained = pca_fitter.explained_variance_ratio_[0]
    logging.info(f"Found a principal component that explains {var_explained} of the variance.")

    # To Neo4j
    logging.info(f"Importing into Neo4j...")
    with driver.session() as session:
        for i, (_, row) in enumerate(df.iterrows()):
            logging.info("  Artist number %d out of %d", i + 1, len(df))
            query = f"""
                MATCH (n:Artist {{main_id: '{row["main_id"]}'}})
                SET
                    n.listeners_scaled = {row["listeners_scaled"]},
                    n.playcount_scaled = {row["playcount_scaled"]},
                    n.popularity = {row["popularity"]},
                    n.popularity_scaled = {row["popularity_scaled"]}
            """
            logging.info("  Artist with main_id '%s'", row["main_id"])
            _ = session.run(query)  # type: ignore


def artist_tags_dates(driver: Driver) -> None:
    query = """
MATCH (n:Artist)
OPTIONAL MATCH (n)-[:HAS_TAG]->(t:Tag)
WITH
    n,
    collect(t.name) AS T0
OPTIONAL MATCH (n)-[:WORKED_IN]->(:Track)-[:HAS_TAG]->(t:Tag)
WITH
    n,
    T0,
    collect(t.name) AS T1
WITH
    n,
    T0 + T1 AS TAGS
SET n.tags = TAGS
WITH
    n,
    TAGS
UNWIND TAGS AS tag
WITH 
    n, 
    tag, 
    count(tag) AS tag_count
ORDER BY n.main_id, tag_count DESC
WITH 
    n,
    min([date IN n.begin_dates WHERE date <> "" | toInteger(date)])[0] AS begin_date,
    max([date IN n.end_dates WHERE date <> "" | toInteger(date)])[0] AS end_date,
    head(collect(tag)) AS most_frequent_tag
SET
    n.begin_date = begin_date,
    n.end_date = end_date,
    n.main_tag = most_frequent_tag
    """
    logging.info("Query:\n%s", query)
    with driver.session() as session:
        _ = session.run(query)
    logging.info("Done")


def do_artists(driver: Driver) -> None:
    logging.info("Artist popularity...")
    artist_popularity(driver)
    logging.info("Tags and dates...")
    artist_tags_dates(driver)


async def track_popularity(asyncdriver: AsyncDriver) -> None:
    # Pandas
    logging.info("Querying and building the main dataframe...")
    async with asyncdriver.session() as session:
        query = """
            MATCH (n:Track)
            RETURN
                n.id AS id,
                n.listeners AS listeners,
                n.playcount AS playcount,
                n.in_last_fm AS in_last_fm
            ;
        """
        q_result = await session.run(query)
        result_data = await q_result.data()
        df = pd.DataFrame(result_data)
    logging.info(f"Found {len(df)} tracks")

    # df with tracks that have info
    logging.info(f"Found {df["in_last_fm"].sum()} tracks with popularity information.")

    # Corr
    r, p_value = pearsonr(df.listeners, df.playcount)
    logging.info(f"Correlation between listeners and playcount: {r}.")
    logging.info(f"Correlation p_value: {p_value}. Significative? {p_value < 0.05}")  # type: ignore

    # Scaling
    mmscaler = MinMaxScaler()
    df["listeners_scaled"] = mmscaler.fit_transform(df[["listeners"]])
    df["playcount_scaled"] = mmscaler.fit_transform(df[["playcount"]])

    # PCA
    pca_fitter = PCA(1)
    pca = pca_fitter.fit_transform(df[["listeners_scaled", "playcount_scaled"]])
    df["popularity"] = pca
    pca_scaled = mmscaler.fit_transform(pca)
    df["popularity_scaled"] = pca_scaled

    var_explained = pca_fitter.explained_variance_ratio_[0]
    logging.info(f"Found a principal component that explains {var_explained} of the variance.")

    # To Neo4j
    logging.info(f"Importing into Neo4j...")

    async def update_batch(asyncdriver: AsyncDriver, batch: tuple[dict]):
        async with asyncdriver.session() as session:
            query = """
                UNWIND $rows as row
                MATCH (n:Track {id: row.id})
                SET
                    n.listeners_scaled = row.listeners_scaled,
                    n.playcount_scaled = row.playcount_scaled,
                    n.popularity = row.popularity,
                    n.popularity_scaled = row.popularity_scaled
                ;
            """
            _ = await session.run(query, rows=batch)

    batch_size = 1000
    batches_per_run = 100
    total_tracks_done = 0
    tasks = list()
    for batch in batched(df.to_dict(orient="records"), batch_size):
        logging.info("Adding a batch to the list")
        tasks.append(update_batch(asyncdriver, batch))  # type: ignore
        logging.info("Batch number %d", len(tasks))

        if len(tasks) >= batches_per_run:
            total_tracks_done += batches_per_run * batch_size
            logging.info("Cleaning batches... %d tracks done after it's done", total_tracks_done)
            await asyncio.gather(*tasks)
            tasks = list()
            logging.info("Tasks are now empty")

    if len(tasks) > 0:
        logging.info("Cleaning final tasks...")
        await asyncio.gather(*tasks)

    logging.info("Done!")


def do_tracks(asyncdriver: AsyncDriver) -> None:
    logging.info("Track popularity...")
    asyncio.run(track_popularity(asyncdriver))


def main(driver: Driver, asyncdriver: AsyncDriver) -> None:
    logging.info("Modifying artists...")
    do_artists(driver)
    logging.info("Modifying tracks...")
    do_tracks(asyncdriver)


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
    asyncdriver = AsyncGraphDatabase.driver(f"bolt://{DB_HOST}:{DB_PORT}", auth=basic_auth(DB_USER, DB_PASS))

    main(driver, asyncdriver)

    driver.close()

    logging.info("DONE!")
