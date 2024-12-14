#!/usr/bin/env python3


from neo4j import GraphDatabase, basic_auth, Driver
from dotenv import load_dotenv
import os
import logging
import pandas as pd


def build_df(driver: Driver) -> pd.DataFrame:
    with driver.session() as session:
        query = """
            MATCH (n:Atrist {in_last_fm: true})
            WITH DISTINCT(n) AS n
            RETURN n.main_id AS main_id, n.popularity_scaled AS popularity
            ORDER BY n.popularity_scaled DESC
            LIMIT 1000
        """
        q_result = session.run(query)


def main(driver: Driver) -> None:
    df = build_df(driver)


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

    logging.info("DONE!")
