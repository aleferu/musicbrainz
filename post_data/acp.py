#!/usr/bin/env python3


from neo4j import GraphDatabase, basic_auth
from dotenv import load_dotenv
import pandas as pd
from sklearn.decomposition import PCA
import os
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import logging


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

    # Pandas
    logging.info("Querying and building the dataframe...")
    with driver.session() as session:
        query = """
            MATCH (n:Artist {in_last_fm: true})
            RETURN n.main_id as main_id, n.listeners as listeners, n.playcount as playcount
        """
        q_result = session.run(query)
        df = pd.DataFrame(q_result.data())
    logging.info(f"Found {len(df)} artists")
    r, p_value = pearsonr(df.listeners, df.playcount)
    logging.info(f"Correlation between listeners and playcount: {r}.")
    logging.info(f"Correlation p_value: {p_value}. Significative? {p_value < 0.05}")  # type: ignore

    # Scaling and PCA
    mmscaler = MinMaxScaler()
    df["listeners_scaled"] = mmscaler.fit_transform(df[["listeners"]])
    df["playcount_scaled"] = mmscaler.fit_transform(df[["playcount"]])

    pca_fitter = PCA(1)
    pca = pca_fitter.fit_transform(df[["listeners_scaled", "playcount_scaled"]])
    df["popularity"] = pca
    pca_scaled = mmscaler.fit_transform(pca)
    df["popularity_scaled"] = pca_scaled

    var_explained = pca_fitter.explained_variance_ratio_[0]
    logging.info(f"Found a principal component that explains {var_explained} of the variance.")
    listeners_component, playcount_component = pca_fitter.components_.flatten()
    logging.info(f"Formula: {listeners_component}listeners + {playcount_component}popularity .")

    fig_path = "img/listeners_playcount.png"
    logging.info(f"Showing results in {fig_path}")
    plt.figure(figsize=(10, 10))
    plt.scatter(df.listeners_scaled, df.playcount_scaled, color="blue", marker="x", label="Data")
    plt.plot(
        df.listeners_scaled,
        (listeners_component / playcount_component) * (df.listeners_scaled - df.listeners_scaled.mean()),
        color="red",
        label="First Principal Component"
    )
    plt.legend()
    plt.title("Listeners vs Playcount")
    plt.xlabel("Listeners")
    plt.ylabel("Playcount")
    plt.savefig(fig_path)

    # Graph
    logging.info("Generating graph...")
    G = nx.Graph()
    # Nodes
    with driver.session() as session:
        query = "MATCH (n:Artist {in_last_fm:true}) RETURN n.main_id as main_id;"
        records = session.run(query)
        for artist in records:
            G.add_node(artist["main_id"])
    # Edges
    with driver.session() as session:
        query = """
            MATCH (a:Artist {in_last_fm: true})-[r:COLLAB_WITH]->(b:Artist {in_last_fm: true})
            WHERE a < b
            RETURN a.main_id as a, b.main_id as b, r.count as weight;
        """
        records = session.run(query)
        for rel in records:
            a, b, w = rel["a"], rel["b"], rel["weight"]
            G.add_edge(a, b, weight=float(w))

    # PageRanks
    logging.info("Computing pageranks...")
    for percentile in [0, 20, 50, 75, 90, 95, 99]:
        # TODO: do the thing for each top (100 - percentile)%
    for lp, hp in [(20, 30), (40, 60), (80, 90)]:
        # TODO: do the thing for each range
