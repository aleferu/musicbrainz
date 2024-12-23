#!/usr/bin/env python3


from neo4j import GraphDatabase, basic_auth, Driver
from dotenv import load_dotenv
import os
import logging
from itertools import combinations_with_replacement
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main(driver: Driver) -> None:
    # DF
    logging.info("Getting distinct tags...")
    query = """
        MATCH (n:Artist)
        WITH COLLECT(DISTINCT n.main_tag) AS all_tags
        RETURN all_tags;
    """
    with driver.session() as session:
        result = session.run(query)
        all_tags = result.data()[0]["all_tags"]
        logging.info("Got: %s", ", ".join(all_tags))

    heatmap_data = pd.DataFrame(
        index=all_tags,
        columns=all_tags,
        dtype=int
    ).fillna(0).astype(int)

    # Other
    with driver.session() as session:
        for t0, t1 in combinations_with_replacement(all_tags, 2):
            logging.info("Querying for '%s' and '%s'...", t0, t1)
            query = f"""
                MATCH (n:Artist {{main_tag: "{t0}"}})-[r:COLLAB_WITH]->(m:Artist {{main_tag: "{t1}"}})
                WHERE n < m
                WITH COUNT(r) AS c
                RETURN toInteger(c) AS c;
            """
            result = session.run(query)  # type: ignore
            count = result.data()[0]["c"]

            logging.info("Got: %d", count)
            heatmap_data.loc[t0, t1] = count
            heatmap_data.loc[t1, t0] = count

    # Drawing
    logging.info("Generating png...")
    plt.figure(figsize=(22, 22))
    sns.heatmap(
        heatmap_data,
        annot=True,
        cmap="coolwarm",
        xticklabels=True,
        yticklabels=True,
        cbar=True,
        fmt="d"
    )
    plt.title("Collaboration between tags")
    plt.savefig("img/Coll_Tags.png")


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