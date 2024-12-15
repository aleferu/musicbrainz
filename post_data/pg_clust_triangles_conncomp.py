#!/usr/bin/env python3


from neo4j import GraphDatabase, basic_auth, Driver
from dotenv import load_dotenv
import pandas as pd
from sklearn.decomposition import PCA
import os
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import logging
import numpy as np
from collections import Counter


def apply_custom_cumsum(df: pd.DataFrame) -> None:
    i = 0
    current_cum_sum = 0
    while i < len(df) - 1:
        if df.loc[i, "pagerank"] != df.loc[i + 1, "pagerank"]:
            current_cum_sum += df.loc[i, "pagerank"]
            df.loc[i, "pagerank_acc"] = current_cum_sum
            i += 1
        else:
            j = i + 1
            current_cum_sum += df.loc[i, "pagerank"]
            while df.loc[j, "pagerank"] == df.loc[i, "pagerank"] and  j < len(df) - 1:
                current_cum_sum += df.loc[j, "pagerank"]
                j += 1
            for k in range(i, j):
                df.loc[k, "pagerank_acc"] = current_cum_sum
            i = j
    df.loc[i, "pagerank_acc"] = current_cum_sum + df.loc[i, "pagerank"]


def conn_comp_thingy(G: nx.Graph) -> None:
    logging.info("        Computing number of Connected Components...")
    number = nx.number_connected_components(G)
    logging.info(f"            Number of Connected Components: {number}")


def clust_coef_thingy(df: pd.DataFrame, G: nx.Graph) -> None:
    logging.info("        Computing Clustering...")
    clusts = nx.clustering(G)
    df["clustering"] = df["main_id"].map(clusts)

    logging.info(f"            Mean: {df["clustering"].mean()}")
    logging.info(f"              Sd: {df["clustering"].std()}")
    logging.info(f"             Min: {df["clustering"].min()}")
    logging.info(f"              Q1: {np.percentile(df["clustering"], 25)}")
    logging.info(f"              Q2: {np.percentile(df["clustering"], 50)}")
    logging.info(f"              Q3: {np.percentile(df["clustering"], 75)}")
    logging.info(f"             Max: {df["clustering"].max()}")

    logging.info("        Computing Triangles...")
    triangles = nx.triangles(G)
    df["triangles"] = df["main_id"].map(triangles)

    logging.info(f"            Mean: {df["triangles"].mean()}")
    logging.info(f"              Sd: {df["triangles"].std()}")
    logging.info(f"             Min: {df["triangles"].min()}")
    logging.info(f"              Q1: {np.percentile(df["triangles"], 25)}")
    logging.info(f"              Q2: {np.percentile(df["triangles"], 50)}")
    logging.info(f"              Q3: {np.percentile(df["triangles"], 75)}")
    logging.info(f"             Max: {df["triangles"].max()}")


def pg_thingy(df: pd.DataFrame, G: nx.Graph):
        logging.info("        Computing PageRank...")
        pg = nx.pagerank(G)
        df["pagerank"] = df["main_id"].map(pg)
        df["pagerank_cumsum"] = df['pagerank'].cumsum()
        df["pagerank_acc"] = df['pagerank_cumsum']  # Need a placeholder

        logging.info("        Computing Accumulated PageRank...")
        apply_custom_cumsum(df)


def print_corrs(df: pd.DataFrame):
    pop_pg, p_pop_pg = pearsonr(df["popularity_scaled"], df["pagerank"])
    pop_pg_cs, p_pop_pg_cs = pearsonr(df["popularity_scaled"], df["pagerank_cumsum"])
    pop_pg_acc, p_pop_pg_acc = pearsonr(df["popularity_scaled"], df["pagerank_acc"])

    pop_cl, p_pop_cl = pearsonr(df["popularity_scaled"], df["clustering"])
    pop_tr, p_pop_tr = pearsonr(df["popularity_scaled"], df["triangles"])

    logging.info("        Correlations: ")
    logging.info(f"            Popularity and Pagerank: {pop_pg} with p-value of {p_pop_pg:.3f}.")
    logging.info(f"            Popularity and Pagerank CumSum: {pop_pg_cs} with p-value of {p_pop_pg_cs:.3f}")
    logging.info(f"            Popularity and Pagerank Acc: {pop_pg_acc} with p-value of {p_pop_pg_acc:.3f}")
    logging.info(f"            Popularity and Clustering: {pop_cl} with p-value of {p_pop_cl:.3f}")
    logging.info(f"            Popularity and Triangles: {pop_tr} with p-value of {p_pop_tr:.3f}")


def do_the_things(df: pd.DataFrame, G: nx.Graph, percentile_start: float, percentile_end: float = 100.0) -> None:
        start_p = int(percentile_start / 100.0 * len(df))
        end_p = int(percentile_end / 100.0 * len(df))
        logging.info(f"    Getting {percentile_end - percentile_start}% of data.")
        logging.info(f"    From {percentile_start} percentile to {percentile_end} percentile.")

        subdf = df.loc[start_p:end_p].copy()
        subdf = subdf.reset_index()
        subg = G.subgraph(subdf.main_id)
        logging.info(f"        Found: {len(subdf)} nodes")

        conn_comp_thingy(subg)
        clust_coef_thingy(subdf, subg)
        pg_thingy(subdf, subg)

        print_corrs(subdf)


def main(driver: Driver) -> None:
    # Pandas
    logging.info("Querying and building the dataframe...")
    with driver.session() as session:
        query = """
            MATCH (n:Artist {in_last_fm: true})-[:COLLAB_WITH]->()
            WITH DISTINCT(n) AS n
            OPTIONAL MATCH (n)-[:HAS_TAG]->(t:Tag)
            WITH
                n,
                COLLECT(t.name) AS T0
            OPTIONAL MATCH (n)-[:WORKED_IN]->(:Track)-[:HAS_TAG]->(t:Tag)
            WITH
                n,
                T0,
                COLLECT(t.name) AS T1
            WITH
                n,
                T0 + T1 AS TAGS
            RETURN n.main_id AS main_id, n.listeners AS listeners, n.playcount AS playcount, TAGS as tags
        """
        q_result = session.run(query)
        df = pd.DataFrame(q_result.data())
    logging.info(f"Found {len(df)} artists")
    r, p_value = pearsonr(df.listeners, df.playcount)
    logging.info(f"Correlation between listeners and playcount: {r}.")
    logging.info(f"Correlation p_value: {p_value}. Significative? {p_value < 0.05}")  # type: ignore

    # Scaling
    mmscaler = MinMaxScaler()
    df["listeners_scaled"] = mmscaler.fit_transform(df[["listeners"]])
    df["playcount_scaled"] = mmscaler.fit_transform(df[["playcount"]])

    # Pre-cleaning
    fig_path = "img/listeners_playcount_pre.png"
    logging.info(f"Showing results in {fig_path}")
    plt.figure(figsize=(10, 10))
    plt.scatter(df.listeners_scaled, df.playcount_scaled, color="blue", marker="x", label="Data")
    plt.legend()
    plt.title("Listeners vs Playcount")
    plt.xlabel("Listeners")
    plt.ylabel("Playcount")
    plt.savefig(fig_path)

    # Cleaning
    df = df[df["listeners_scaled"] <= 0.8]
    df = df[df["playcount_scaled"] <= 0.6]
    df["listeners_scaled"] = mmscaler.fit_transform(df[["listeners"]])
    df["playcount_scaled"] = mmscaler.fit_transform(df[["playcount"]])
    assert type(df) == pd.DataFrame, "LSP"

    logging.info(f"Found {len(df)} artists after cleanup.")
    r, p_value = pearsonr(df.listeners, df.playcount)
    logging.info(f"Correlation between listeners and playcount: {r}.")
    logging.info(f"Correlation p_value: {p_value}. Significative? {p_value < 0.05}")  # type: ignore

    # Post-cleaning
    fig_path = "img/listeners_playcount_post.png"
    logging.info(f"Showing results in {fig_path}")
    plt.figure(figsize=(10, 10))
    plt.scatter(df.listeners_scaled, df.playcount_scaled, color="blue", marker="x", label="Data")
    plt.legend()
    plt.title("Listeners vs Playcount")
    plt.xlabel("Listeners")
    plt.ylabel("Playcount")
    plt.savefig(fig_path)

    # Main genre
    df["main_tag"] = df["tags"].map(
        lambda x: Counter(x).most_common(1)[0][0] if x else "No data"
    )

    # PCA
    pca_fitter = PCA(1)
    pca = pca_fitter.fit_transform(df[["listeners_scaled", "playcount_scaled"]])
    df["popularity"] = pca
    pca_scaled = mmscaler.fit_transform(pca)
    df["popularity_scaled"] = pca_scaled

    var_explained = pca_fitter.explained_variance_ratio_[0]
    logging.info(f"Found a principal component that explains {var_explained} of the variance.")

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

    # Sort by 
    df.sort_values(by="popularity_scaled", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Stats in general
    logging.info("Computing stats for all the data...")
    for percentile in [0, 20, 50, 75, 90, 95, 99, 99.5, 99.9]:
        do_the_things(df, G, percentile, 100)

    ranges = [
        (0, 5),
        (5, 10),
        (10, 15),
        (20, 30),
        (35, 45),
        (60, 70),
        (80, 90),
        (90, 95),
        (95, 99),
        (95, 100),
        (97, 99),
        (99, 99.5),
        (99.5, 99.7),
        (99.8, 99.9),
        (99.9, 99.95)
    ]
    for lp, hp in ranges:
        do_the_things(df, G, lp, hp)


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
