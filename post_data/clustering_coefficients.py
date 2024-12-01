#!/usr/bin/env python3


import os
from neo4j import GraphDatabase, basic_auth
import networkx as nx
import logging
from dotenv import load_dotenv
import numpy as np
from functools import reduce


# https://networkx.org/documentation/stable/reference/algorithms/clustering.html


# Sample output:
"""
2024-12-01 18:44:54 - INFO - Connection established! Generating graphs...
2024-12-01 18:45:26 - INFO - Done! Found 593081 nodes and 1231526 edges.
2024-12-01 18:45:26 - INFO - ------------------------------------------
2024-12-01 18:45:26 - INFO - Computing average clustering...
2024-12-01 18:45:37 - INFO - Computing all clustering coefficients...
2024-12-01 18:45:48 - INFO - Got average clustering of 0.2624356267622425
2024-12-01 18:45:48 - INFO - Mean: 0.2624356267622424
2024-12-01 18:45:48 - INFO -   Sd: 0.3923628665542284
2024-12-01 18:45:48 - INFO -  Min: 0.0
2024-12-01 18:45:48 - INFO -   Q1: 0.0
2024-12-01 18:45:48 - INFO -   Q2: 0.0
2024-12-01 18:45:48 - INFO -   Q3: 0.4
2024-12-01 18:45:48 - INFO -  Max: 1.0
2024-12-01 18:45:48 - INFO - ------------------------------------------
2024-12-01 18:45:48 - INFO - Computing number of triangles...
2024-12-01 18:45:52 - INFO - Got triangle count of 3038412
2024-12-01 18:45:52 - INFO - Mean: 5.123097856785161
2024-12-01 18:45:52 - INFO -   Sd: 63.90716559331751
2024-12-01 18:45:52 - INFO -  Min: 0
2024-12-01 18:45:52 - INFO -   Q1: 0.0
2024-12-01 18:45:52 - INFO -   Q2: 0.0
2024-12-01 18:45:52 - INFO -   Q3: 2.0
2024-12-01 18:45:52 - INFO -  Max: 11266
2024-12-01 18:45:52 - INFO - Cleaning...
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

    logging.info("Connection established! Generating graphs...")

    # Data gathering
    G = nx.Graph()
    query = "MATCH (n:Artist)-[:COLLAB_WITH]->(m:Artist) WHERE n < m RETURN n.main_id as a, m.main_id AS b"
    with driver.session() as session:
        q_result = session.run(query)
        for r in q_result:
            G.add_edge(r["a"], r["b"])
    logging.info(f"Done! Found {len(G.nodes)} nodes and {len(G.edges)} edges.")

    logging.info("------------------------------------------")

    # Computations
    logging.info(f"Computing average clustering...")
    avg_clust = nx.average_clustering(G, count_zeros=True)

    logging.info(f"Computing all clustering coefficients...")
    clust = np.array([c for c in nx.clustering(G).values()])

    # Results
    logging.info(f"Got average clustering of {avg_clust}")
    logging.info(f"Mean: {clust.mean()}")
    logging.info(f"  Sd: {clust.std()}")
    logging.info(f" Min: {clust.min()}")
    logging.info(f"  Q1: {np.percentile(clust, 25)}")
    logging.info(f"  Q2: {np.percentile(clust, 50)}")
    logging.info(f"  Q3: {np.percentile(clust, 75)}")
    logging.info(f" Max: {clust.max()}")

    logging.info("------------------------------------------")

    # More computations
    logging.info(f"Computing number of triangles...")
    triangle_count = np.array([t for t in nx.triangles(G, nodes=None).values()])

    # More results
    logging.info(f"Got triangle count of {triangle_count.sum()}")
    logging.info(f"Mean: {triangle_count.mean()}")
    logging.info(f"  Sd: {triangle_count.std()}")
    logging.info(f" Min: {triangle_count.min()}")
    logging.info(f"  Q1: {np.percentile(triangle_count, 25)}")
    logging.info(f"  Q2: {np.percentile(triangle_count, 50)}")
    logging.info(f"  Q3: {np.percentile(triangle_count, 75)}")
    logging.info(f" Max: {triangle_count.max()}")

    # cleanup
    logging.info("Cleaning...")
    driver.close()
