#!/usr/bin/env python3


import logging
from dotenv import load_dotenv
import os
from neo4j import GraphDatabase, basic_auth
import networkx as nx


# https://networkx.org/documentation/stable/reference/algorithms/component.html


# Sample output:
"""
2024-12-01 19:16:12 - INFO - Connection established! Generating graph...
2024-12-01 19:17:16 - INFO - Done! Found 593081 nodes and 2463052 edges.
2024-12-01 19:17:16 - INFO - ------------------------------------------
2024-12-01 19:17:16 - INFO - Computing connections...
2024-12-01 19:17:21 - INFO - 55148
2024-12-01 19:17:21 - INFO - Cleaning...
"""


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

    logging.info("Connection established! Generating graph...")

    # Data gathering
    G = nx.DiGraph()
    query = "MATCH (n:Artist)-[:COLLAB_WITH]->(m:Artist) RETURN n.main_id as a, m.main_id AS b"
    with driver.session() as session:
        q_result = session.run(query)
        for r in q_result:
            G.add_edge(r["a"], r["b"])
    logging.info(f"Done! Found {len(G.nodes)} nodes and {len(G.edges)} edges.")

    logging.info("------------------------------------------")

    # Computations
    logging.info(f"Computing connections...")
    logging.info(len([foo for foo in nx.strongly_connected_components(G)]))

    # cleanup
    logging.info("Cleaning...")
    driver.close()
