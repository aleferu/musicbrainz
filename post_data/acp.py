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


# Sample output:

"""
2024-12-08 20:35:21 - INFO - Querying and building the dataframe...
2024-12-08 20:35:39 - INFO - Found 512749 artists
2024-12-08 20:35:39 - INFO - Correlation between listeners and playcount: 0.6955042178755177.
2024-12-08 20:35:39 - INFO - Correlation p_value: 0.0. Significative? True
2024-12-08 20:35:39 - INFO - Showing results in img/listeners_playcount_pre.png
2024-12-08 20:35:40 - INFO - Found 512746 artists after cleanup.
2024-12-08 20:35:40 - INFO - Correlation between listeners and playcount: 0.7597527490635692.
2024-12-08 20:35:40 - INFO - Correlation p_value: 0.0. Significative? True
2024-12-08 20:35:40 - INFO - Showing results in img/listeners_playcount_post.png
2024-12-08 20:35:40 - INFO - Found a principal component that explains 0.9487295349587309 of the variance.
2024-12-08 20:35:40 - INFO - Generating graph...
2024-12-08 20:36:40 - INFO - Computing pageranks...
2024-12-08 20:36:40 - INFO -     Getting 100% of data.
2024-12-08 20:36:40 - INFO -     From 0 percentile to 100 percentile.
2024-12-08 20:36:40 - INFO -         Found: 512746 nodes
2024-12-08 20:36:40 - INFO -         Computing PageRank...
2024-12-08 20:37:48 - INFO -         Correlations: 
2024-12-08 20:37:48 - INFO -             Popularity and Pagerank: 0.1789509894864647 with p-value of 0.000.
2024-12-08 20:37:48 - INFO -             Popularity and Pagerank CumSum: 0.2909698476642479 with p-value of 0.000
2024-12-08 20:37:48 - INFO -             Popularity and Pagerank Acc: 0.2909698538125427 with p-value of 0.000
2024-12-08 20:37:48 - INFO -     Getting 80% of data.
2024-12-08 20:37:48 - INFO -     From 20 percentile to 100 percentile.
2024-12-08 20:37:48 - INFO -         Found: 410197 nodes
2024-12-08 20:37:48 - INFO -         Computing PageRank...
2024-12-08 20:38:43 - INFO -         Correlations: 
2024-12-08 20:38:43 - INFO -             Popularity and Pagerank: 0.1820800644847153 with p-value of 0.000.
2024-12-08 20:38:43 - INFO -             Popularity and Pagerank CumSum: 0.3188585581712925 with p-value of 0.000
2024-12-08 20:38:43 - INFO -             Popularity and Pagerank Acc: 0.3188585647317288 with p-value of 0.000
2024-12-08 20:38:43 - INFO -     Getting 50% of data.
2024-12-08 20:38:43 - INFO -     From 50 percentile to 100 percentile.
2024-12-08 20:38:43 - INFO -         Found: 256373 nodes
2024-12-08 20:38:43 - INFO -         Computing PageRank...
2024-12-08 20:39:18 - INFO -         Correlations: 
2024-12-08 20:39:18 - INFO -             Popularity and Pagerank: 0.19054209701520758 with p-value of 0.000.
2024-12-08 20:39:18 - INFO -             Popularity and Pagerank CumSum: 0.37897596294115093 with p-value of 0.000
2024-12-08 20:39:18 - INFO -             Popularity and Pagerank Acc: 0.3789759722429687 with p-value of 0.000
2024-12-08 20:39:18 - INFO -     Getting 25% of data.
2024-12-08 20:39:18 - INFO -     From 75 percentile to 100 percentile.
2024-12-08 20:39:18 - INFO -         Found: 128187 nodes
2024-12-08 20:39:18 - INFO -         Computing PageRank...
2024-12-08 20:39:35 - INFO -         Correlations: 
2024-12-08 20:39:35 - INFO -             Popularity and Pagerank: 0.1947504953710363 with p-value of 0.000.
2024-12-08 20:39:35 - INFO -             Popularity and Pagerank CumSum: 0.47366965641884445 with p-value of 0.000
2024-12-08 20:39:35 - INFO -             Popularity and Pagerank Acc: 0.4736696784746938 with p-value of 0.000
2024-12-08 20:39:35 - INFO -     Getting 10% of data.
2024-12-08 20:39:35 - INFO -     From 90 percentile to 100 percentile.
2024-12-08 20:39:35 - INFO -         Found: 51275 nodes
2024-12-08 20:39:35 - INFO -         Computing PageRank...
2024-12-08 20:39:42 - INFO -         Correlations: 
2024-12-08 20:39:42 - INFO -             Popularity and Pagerank: 0.20596590725184724 with p-value of 0.000.
2024-12-08 20:39:42 - INFO -             Popularity and Pagerank CumSum: 0.5931906878005918 with p-value of 0.000
2024-12-08 20:39:42 - INFO -             Popularity and Pagerank Acc: 0.5931907741262822 with p-value of 0.000
2024-12-08 20:39:42 - INFO -     Getting 5% of data.
2024-12-08 20:39:42 - INFO -     From 95 percentile to 100 percentile.
2024-12-08 20:39:42 - INFO -         Found: 25638 nodes
2024-12-08 20:39:42 - INFO -         Computing PageRank...
2024-12-08 20:39:45 - INFO -         Correlations: 
2024-12-08 20:39:45 - INFO -             Popularity and Pagerank: 0.20622980501331128 with p-value of 0.000.
2024-12-08 20:39:45 - INFO -             Popularity and Pagerank CumSum: 0.6826799920932141 with p-value of 0.000
2024-12-08 20:39:45 - INFO -             Popularity and Pagerank Acc: 0.6826806212399116 with p-value of 0.000
2024-12-08 20:39:45 - INFO -     Getting 1% of data.
2024-12-08 20:39:45 - INFO -     From 99 percentile to 100 percentile.
2024-12-08 20:39:45 - INFO -         Found: 5128 nodes
2024-12-08 20:39:45 - INFO -         Computing PageRank...
2024-12-08 20:39:46 - INFO -         Correlations: 
2024-12-08 20:39:46 - INFO -             Popularity and Pagerank: 0.17840308795260806 with p-value of 0.000.
2024-12-08 20:39:46 - INFO -             Popularity and Pagerank CumSum: 0.8545487623944918 with p-value of 0.000
2024-12-08 20:39:46 - INFO -             Popularity and Pagerank Acc: 0.8545519286985351 with p-value of 0.000
2024-12-08 20:39:46 - INFO -     Getting 0.5% of data.
2024-12-08 20:39:46 - INFO -     From 99.5 percentile to 100 percentile.
2024-12-08 20:39:46 - INFO -         Found: 2564 nodes
2024-12-08 20:39:46 - INFO -         Computing PageRank...
2024-12-08 20:39:46 - INFO -         Correlations: 
2024-12-08 20:39:46 - INFO -             Popularity and Pagerank: 0.15196810956379872 with p-value of 0.000.
2024-12-08 20:39:46 - INFO -             Popularity and Pagerank CumSum: 0.8871432010644135 with p-value of 0.000
2024-12-08 20:39:46 - INFO -             Popularity and Pagerank Acc: 0.8871349008474978 with p-value of 0.000
2024-12-08 20:39:46 - INFO -     Getting 0.09999999999999432% of data.
2024-12-08 20:39:46 - INFO -     From 99.9 percentile to 100 percentile.
2024-12-08 20:39:46 - INFO -         Found: 513 nodes
2024-12-08 20:39:46 - INFO -         Computing PageRank...
2024-12-08 20:39:47 - INFO -         Correlations: 
2024-12-08 20:39:47 - INFO -             Popularity and Pagerank: 0.11171684925299483 with p-value of 0.011.
2024-12-08 20:39:47 - INFO -             Popularity and Pagerank CumSum: 0.9081292697620668 with p-value of 0.000
2024-12-08 20:39:47 - INFO -             Popularity and Pagerank Acc: 0.9085811829703403 with p-value of 0.000
2024-12-08 20:39:47 - INFO -     Getting 5% of data.
2024-12-08 20:39:47 - INFO -     From 0 percentile to 5 percentile.
2024-12-08 20:39:47 - INFO -         Found: 25638 nodes
2024-12-08 20:39:47 - INFO -         Computing PageRank...
2024-12-08 20:39:49 - INFO -         Correlations: 
2024-12-08 20:39:49 - INFO -             Popularity and Pagerank: -0.05875880449982209 with p-value of 0.000.
2024-12-08 20:39:49 - INFO -             Popularity and Pagerank CumSum: 0.6919917299493559 with p-value of 0.000
2024-12-08 20:39:49 - INFO -             Popularity and Pagerank Acc: 0.6921041621098396 with p-value of 0.000
2024-12-08 20:39:49 - INFO -     Getting 5% of data.
2024-12-08 20:39:49 - INFO -     From 5 percentile to 10 percentile.
2024-12-08 20:39:49 - INFO -         Found: 25638 nodes
2024-12-08 20:39:49 - INFO -         Computing PageRank...
2024-12-08 20:39:52 - INFO -         Correlations: 
2024-12-08 20:39:52 - INFO -             Popularity and Pagerank: -0.0038475535277358534 with p-value of 0.538.
2024-12-08 20:39:52 - INFO -             Popularity and Pagerank CumSum: 0.9283206594517351 with p-value of 0.000
2024-12-08 20:39:52 - INFO -             Popularity and Pagerank Acc: 0.9283531797894087 with p-value of 0.000
2024-12-08 20:39:52 - INFO -     Getting 5% of data.
2024-12-08 20:39:52 - INFO -     From 10 percentile to 15 percentile.
2024-12-08 20:39:52 - INFO -         Found: 25638 nodes
2024-12-08 20:39:52 - INFO -         Computing PageRank...
2024-12-08 20:39:55 - INFO -         Correlations: 
2024-12-08 20:39:55 - INFO -             Popularity and Pagerank: 0.0041880554612424275 with p-value of 0.503.
2024-12-08 20:39:55 - INFO -             Popularity and Pagerank CumSum: 0.9876850742963205 with p-value of 0.000
2024-12-08 20:39:55 - INFO -             Popularity and Pagerank Acc: 0.9876917388498013 with p-value of 0.000
2024-12-08 20:39:55 - INFO -     Getting 10% of data.
2024-12-08 20:39:55 - INFO -     From 20 percentile to 30 percentile.
2024-12-08 20:39:55 - INFO -         Found: 51275 nodes
2024-12-08 20:39:55 - INFO -         Computing PageRank...
2024-12-08 20:40:01 - INFO -         Correlations: 
2024-12-08 20:40:01 - INFO -             Popularity and Pagerank: 0.0073791247005198 with p-value of 0.095.
2024-12-08 20:40:01 - INFO -             Popularity and Pagerank CumSum: 0.9925210374806782 with p-value of 0.000
2024-12-08 20:40:01 - INFO -             Popularity and Pagerank Acc: 0.9925213517468874 with p-value of 0.000
2024-12-08 20:40:01 - INFO -     Getting 10% of data.
2024-12-08 20:40:01 - INFO -     From 35 percentile to 45 percentile.
2024-12-08 20:40:01 - INFO -         Found: 51275 nodes
2024-12-08 20:40:01 - INFO -         Computing PageRank...
2024-12-08 20:40:07 - INFO -         Correlations: 
2024-12-08 20:40:07 - INFO -             Popularity and Pagerank: 0.005807284773996691 with p-value of 0.189.
2024-12-08 20:40:07 - INFO -             Popularity and Pagerank CumSum: 0.9937776578974294 with p-value of 0.000
2024-12-08 20:40:07 - INFO -             Popularity and Pagerank Acc: 0.9937778198710978 with p-value of 0.000
2024-12-08 20:40:07 - INFO -     Getting 10% of data.
2024-12-08 20:40:07 - INFO -     From 60 percentile to 70 percentile.
2024-12-08 20:40:07 - INFO -         Found: 51276 nodes
2024-12-08 20:40:07 - INFO -         Computing PageRank...
2024-12-08 20:40:13 - INFO -         Correlations: 
2024-12-08 20:40:13 - INFO -             Popularity and Pagerank: 0.025348873767087932 with p-value of 0.000.
2024-12-08 20:40:13 - INFO -             Popularity and Pagerank CumSum: 0.9923438642225452 with p-value of 0.000
2024-12-08 20:40:13 - INFO -             Popularity and Pagerank Acc: 0.9923438698103664 with p-value of 0.000
2024-12-08 20:40:13 - INFO -     Getting 10% of data.
2024-12-08 20:40:13 - INFO -     From 80 percentile to 90 percentile.
2024-12-08 20:40:13 - INFO -         Found: 51276 nodes
2024-12-08 20:40:13 - INFO -         Computing PageRank...
2024-12-08 20:40:19 - INFO -         Correlations: 
2024-12-08 20:40:19 - INFO -             Popularity and Pagerank: 0.05810397252345869 with p-value of 0.000.
2024-12-08 20:40:19 - INFO -             Popularity and Pagerank CumSum: 0.9830173013411414 with p-value of 0.000
2024-12-08 20:40:19 - INFO -             Popularity and Pagerank Acc: 0.9830172924303126 with p-value of 0.000
2024-12-08 20:40:19 - INFO -     Getting 5% of data.
2024-12-08 20:40:19 - INFO -     From 90 percentile to 95 percentile.
2024-12-08 20:40:19 - INFO -         Found: 25638 nodes
2024-12-08 20:40:19 - INFO -         Computing PageRank...
2024-12-08 20:40:22 - INFO -         Correlations: 
2024-12-08 20:40:22 - INFO -             Popularity and Pagerank: 0.05936054304711645 with p-value of 0.000.
2024-12-08 20:40:22 - INFO -             Popularity and Pagerank CumSum: 0.9888972112126172 with p-value of 0.000
2024-12-08 20:40:22 - INFO -             Popularity and Pagerank Acc: 0.9888973559511224 with p-value of 0.000
2024-12-08 20:40:22 - INFO -     Getting 4% of data.
2024-12-08 20:40:22 - INFO -     From 95 percentile to 99 percentile.
2024-12-08 20:40:22 - INFO -         Found: 20511 nodes
2024-12-08 20:40:22 - INFO -         Computing PageRank...
2024-12-08 20:40:25 - INFO -         Correlations: 
2024-12-08 20:40:25 - INFO -             Popularity and Pagerank: 0.10716521311152152 with p-value of 0.000.
2024-12-08 20:40:25 - INFO -             Popularity and Pagerank CumSum: 0.9472264970251942 with p-value of 0.000
2024-12-08 20:40:25 - INFO -             Popularity and Pagerank Acc: 0.9472264954801997 with p-value of 0.000
2024-12-08 20:40:25 - INFO -     Getting 5% of data.
2024-12-08 20:40:25 - INFO -     From 95 percentile to 100 percentile.
2024-12-08 20:40:25 - INFO -         Found: 25638 nodes
2024-12-08 20:40:25 - INFO -         Computing PageRank...
2024-12-08 20:40:28 - INFO -         Correlations: 
2024-12-08 20:40:28 - INFO -             Popularity and Pagerank: 0.20622980501331128 with p-value of 0.000.
2024-12-08 20:40:28 - INFO -             Popularity and Pagerank CumSum: 0.6826799920932141 with p-value of 0.000
2024-12-08 20:40:28 - INFO -             Popularity and Pagerank Acc: 0.6826806212399116 with p-value of 0.000
2024-12-08 20:40:28 - INFO -     Getting 2% of data.
2024-12-08 20:40:28 - INFO -     From 97 percentile to 99 percentile.
2024-12-08 20:40:28 - INFO -         Found: 10256 nodes
2024-12-08 20:40:28 - INFO -         Computing PageRank...
2024-12-08 20:40:29 - INFO -         Correlations: 
2024-12-08 20:40:29 - INFO -             Popularity and Pagerank: 0.09094306474851528 with p-value of 0.000.
2024-12-08 20:40:29 - INFO -             Popularity and Pagerank CumSum: 0.9771174183877903 with p-value of 0.000
2024-12-08 20:40:29 - INFO -             Popularity and Pagerank Acc: 0.9771171400685633 with p-value of 0.000
2024-12-08 20:40:29 - INFO -     Getting 0.5% of data.
2024-12-08 20:40:29 - INFO -     From 99 percentile to 99.5 percentile.
2024-12-08 20:40:29 - INFO -         Found: 2565 nodes
2024-12-08 20:40:29 - INFO -         Computing PageRank...
2024-12-08 20:40:30 - INFO -         Correlations: 
2024-12-08 20:40:30 - INFO -             Popularity and Pagerank: 0.04501996406734586 with p-value of 0.023.
2024-12-08 20:40:30 - INFO -             Popularity and Pagerank CumSum: 0.9911563259149724 with p-value of 0.000
2024-12-08 20:40:30 - INFO -             Popularity and Pagerank Acc: 0.9911585419829672 with p-value of 0.000
2024-12-08 20:40:30 - INFO -     Getting 0.20000000000000284% of data.
2024-12-08 20:40:30 - INFO -     From 99.5 percentile to 99.7 percentile.
2024-12-08 20:40:30 - INFO -         Found: 1026 nodes
2024-12-08 20:40:30 - INFO -         Computing PageRank...
2024-12-08 20:40:30 - INFO -         Correlations: 
2024-12-08 20:40:30 - INFO -             Popularity and Pagerank: 0.03767361497540848 with p-value of 0.228.
2024-12-08 20:40:30 - INFO -             Popularity and Pagerank CumSum: 0.9967418435502823 with p-value of 0.000
2024-12-08 20:40:30 - INFO -             Popularity and Pagerank Acc: 0.9967887793519727 with p-value of 0.000
2024-12-08 20:40:30 - INFO -     Getting 0.10000000000000853% of data.
2024-12-08 20:40:30 - INFO -     From 99.8 percentile to 99.9 percentile.
2024-12-08 20:40:30 - INFO -         Found: 514 nodes
2024-12-08 20:40:30 - INFO -         Computing PageRank...
2024-12-08 20:40:30 - INFO -         Correlations: 
2024-12-08 20:40:30 - INFO -             Popularity and Pagerank: 0.05831251396456604 with p-value of 0.187.
2024-12-08 20:40:30 - INFO -             Popularity and Pagerank CumSum: 0.9974751047463978 with p-value of 0.000
2024-12-08 20:40:30 - INFO -             Popularity and Pagerank Acc: 0.9975320945880538 with p-value of 0.000
2024-12-08 20:40:30 - INFO -     Getting 0.04999999999999716% of data.
2024-12-08 20:40:30 - INFO -     From 99.9 percentile to 99.95 percentile.
2024-12-08 20:40:30 - INFO -         Found: 257 nodes
2024-12-08 20:40:30 - INFO -         Computing PageRank...
2024-12-08 20:40:30 - INFO -         Correlations: 
2024-12-08 20:40:30 - INFO -             Popularity and Pagerank: 0.16745426269344713 with p-value of 0.007.
2024-12-08 20:40:30 - INFO -             Popularity and Pagerank CumSum: 0.9854481438714142 with p-value of 0.000
2024-12-08 20:40:30 - INFO -             Popularity and Pagerank Acc: 0.9849301940149133 with p-value of 0.000
2024-12-08 20:40:30 - INFO - DONE!
"""


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


def clust_coef_thingy(G: nx.Graph) -> None:
    logging.info("        Computing Clustering...")
    clust = np.array([c for c in nx.clustering(G).values()])

    logging.info(f"            Mean: {clust.mean()}")
    logging.info(f"              Sd: {clust.std()}")
    logging.info(f"             Min: {clust.min()}")
    logging.info(f"              Q1: {np.percentile(clust, 25)}")
    logging.info(f"              Q2: {np.percentile(clust, 50)}")
    logging.info(f"              Q3: {np.percentile(clust, 75)}")
    logging.info(f"             Max: {clust.max()}")

    logging.info("        Computing Triangles...")
    triangles = np.array([t for t in nx.triangles(G).values()])

    logging.info(f"            Mean: {triangles.mean()}")
    logging.info(f"              Sd: {triangles.std()}")
    logging.info(f"             Min: {triangles.min()}")
    logging.info(f"              Q1: {np.percentile(triangles, 25)}")
    logging.info(f"              Q2: {np.percentile(triangles, 50)}")
    logging.info(f"              Q3: {np.percentile(triangles, 75)}")
    logging.info(f"             Max: {triangles.max()}")


def pg_thingy(df: pd.DataFrame, G: nx.Graph):
        logging.info("        Computing PageRank...")
        pg = nx.pagerank(G)
        df["pagerank"] = df["main_id"].map(pg)
        df["pagerank_cumsum"] = df['pagerank'].cumsum()
        df["pagerank_acc"] = df['pagerank_cumsum']  # Need a placeholder

        apply_custom_cumsum(df)

        pop_pg, p_pop_pg = pearsonr(df["popularity_scaled"], df["pagerank"])
        pop_pg_cs, p_pop_pg_cs = pearsonr(df["popularity_scaled"], df["pagerank_cumsum"])
        pop_pg_acc, p_pop_pg_acc = pearsonr(df["popularity_scaled"], df["pagerank_acc"])

        logging.info("        Correlations: ")
        logging.info(f"            Popularity and Pagerank: {pop_pg} with p-value of {p_pop_pg:.3f}.")
        logging.info(f"            Popularity and Pagerank CumSum: {pop_pg_cs} with p-value of {p_pop_pg_cs:.3f}")
        logging.info(f"            Popularity and Pagerank Acc: {pop_pg_acc} with p-value of {p_pop_pg_acc:.3f}")


def do_the_things(df: pd.DataFrame, G: nx.Graph, percentile_start: float, percentile_end: float = 100.0) -> None:
        start_p = int(percentile_start / 100.0 * len(df))
        end_p = int(percentile_end / 100.0 * len(df))
        logging.info(f"    Getting {percentile_end - percentile_start}% of data.")
        logging.info(f"    From {percentile_start} percentile to {percentile_end} percentile.")

        subdf = df.loc[start_p:end_p].copy()
        subdf = subdf.reset_index()
        subg = G.subgraph(subdf.main_id)
        logging.info(f"        Found: {len(subdf)} nodes")

        pg_thingy(subdf, subg)
        conn_comp_thingy(subg)
        clust_coef_thingy(subg)


def main(driver: Driver) -> None:
    # Pandas
    logging.info("Querying and building the dataframe...")
    with driver.session() as session:
        query = """
            MATCH (n:Artist {in_last_fm: true})-[:COLLAB_WITH]->()
            WITH DISTINCT(n) AS n
            RETURN n.main_id AS main_id, n.listeners AS listeners, n.playcount AS playcount
        """
        q_result = session.run(query)
        df = pd.DataFrame(q_result.data())
    logging.info(f"Found {len(df)} artists")
    r, p_value = pearsonr(df.listeners, df.playcount)
    logging.info(f"Correlation between listeners and playcount: {r}.")
    logging.info(f"Correlation p_value: {p_value}. Significative? {p_value < 0.05}")  # type: ignore

    # Scaling
    stdscaler = MinMaxScaler()
    df["listeners_scaled"] = stdscaler.fit_transform(df[["listeners"]])
    df["playcount_scaled"] = stdscaler.fit_transform(df[["playcount"]])

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
    df["listeners_scaled"] = stdscaler.fit_transform(df[["listeners"]])
    df["playcount_scaled"] = stdscaler.fit_transform(df[["playcount"]])
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

    # PCA
    pca_fitter = PCA(1)
    pca = pca_fitter.fit_transform(df[["listeners_scaled", "playcount_scaled"]])
    df["popularity"] = pca
    pca_scaled = stdscaler.fit_transform(pca)
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

    # PageRanks
    logging.info("Computing pageranks...")
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

    logging.info("DONE!")
