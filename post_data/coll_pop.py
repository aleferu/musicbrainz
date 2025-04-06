#!/usr/bin/env python3


from typing import Any
from neo4j import GraphDatabase, basic_auth, Driver, Transaction
from dotenv import load_dotenv
import os
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_collaboration_counts(tx: Transaction) -> list[dict[str, Any]]:
    logging.info("Calculating quartile boundaries and collaboration counts...")
    query = """
        MATCH (a:Artist)
        WHERE a.popularity_scaled IS NOT NULL
        WITH percentileDisc(a.popularity_scaled, 0.25) AS q1_bound,
             percentileDisc(a.popularity_scaled, 0.50) AS q2_bound,
             percentileDisc(a.popularity_scaled, 0.75) AS q3_bound

        MATCH (a1:Artist)-[:COLLAB_WITH]-(a2:Artist)
        WHERE a1.popularity_scaled IS NOT NULL AND a2.popularity_scaled IS NOT NULL

        WITH q1_bound, q2_bound, q3_bound, a1, a2,
             CASE
               WHEN a1.popularity_scaled <= q1_bound THEN 'Q1'
               WHEN a1.popularity_scaled <= q2_bound THEN 'Q2'
               WHEN a1.popularity_scaled <= q3_bound THEN 'Q3'
               ELSE 'Q4'
             END AS quartile1,
             CASE
               WHEN a2.popularity_scaled <= q1_bound THEN 'Q1'
               WHEN a2.popularity_scaled <= q2_bound THEN 'Q2'
               WHEN a2.popularity_scaled <= q3_bound THEN 'Q3'
               ELSE 'Q4'
             END AS quartile2

        RETURN quartile1, quartile2, count(*) AS collaboration_count
    """
    result = tx.run(query)
    return result.data()


def main(driver: Driver) -> None:
    index_cols = ["Q1", "Q2", "Q3", "Q4"]
    heatmap_data = pd.DataFrame(
        index=index_cols,  # type: ignore
        columns=index_cols,  # type: ignore
        dtype=int
    ).fillna(0.0).astype(int)

    logging.info("Querying Neo4j for collaboration data...")
    try:
        with driver.session() as session:
            results = session.execute_read(get_collaboration_counts)  # type: ignore

    except Exception as e:
        logging.error(f"Error querying Neo4j: {e}")
        return

    logging.info("Filling up heatmap_data...")
    for row in results:
        q1 = row['quartile1']
        q2 = row['quartile2']
        count = row['collaboration_count']

        logging.info("Got %d for %s and %s", count, q1, q2)
        heatmap_data.loc[q1, q2] = count
        heatmap_data.loc[q2, q1] = count

    # Normalization by row
    row_sums = heatmap_data.sum(axis=1)
    normalized_heatmap_data = heatmap_data.div(row_sums, axis=0)

    # Drawing
    logging.info("Generating png...")

    plt.figure(figsize=(5, 5))
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
    plt.savefig(f"img/coll_pop.png")


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

    main(driver)

    driver.close()

    logging.info("DONE!")
