#!/usr/bin/env python3


from neo4j import GraphDatabase, basic_auth
import pandas as pd
import os
import logging
from dotenv import load_dotenv


def main():
    logging.info("Querying...")
    # year month combos
    driver = GraphDatabase.driver(f"bolt://{DB_HOST}:{DB_PORT}", auth=basic_auth(DB_USER, DB_PASS))  # type: ignore
    with driver.session() as session:
        query = """
            MATCH (n:Track)
            WHERE n.year <> "0"
            WITH
                toInteger(n.year) AS year,
                toInteger(n.month) AS month,
                n.id AS track_id
            RETURN DISTINCT year, month, COLLECT(track_id) AS track_ids
        """
        q_result = session.run(query)  # type: ignore

        # DataFrame
        logging.info("Building DataFrame...")
        df = pd.DataFrame(q_result.data())

    logging.info("Saving...")
    df.to_csv("data/year_month_track.csv", index=False)

    logging.info("Done!")


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

    main()
