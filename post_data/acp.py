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
2024-12-14 22:56:21 - INFO - Querying and building the dataframe...
2024-12-14 22:56:40 - INFO - Found 512749 artists
2024-12-14 22:56:40 - INFO - Correlation between listeners and playcount: 0.6955042178755177.
2024-12-14 22:56:40 - INFO - Correlation p_value: 0.0. Significative? True
2024-12-14 22:56:40 - INFO - Showing results in img/listeners_playcount_pre.png
2024-12-14 22:56:41 - INFO - Found 512746 artists after cleanup.
2024-12-14 22:56:41 - INFO - Correlation between listeners and playcount: 0.7597527490635692.
2024-12-14 22:56:41 - INFO - Correlation p_value: 0.0. Significative? True
2024-12-14 22:56:41 - INFO - Showing results in img/listeners_playcount_post.png
2024-12-14 22:56:42 - INFO - Found a principal component that explains 0.9487295349587309 of the variance.
2024-12-14 22:56:42 - INFO - Generating graph...
2024-12-14 22:57:42 - INFO - Computing pageranks...
2024-12-14 22:57:42 - INFO -     Getting 100% of data.
2024-12-14 22:57:42 - INFO -     From 0 percentile to 100 percentile.
2024-12-14 22:57:42 - INFO -         Found: 512746 nodes
2024-12-14 22:57:42 - INFO -         Computing number of Connected Components...
2024-12-14 22:57:45 - INFO -             Number of Connected Components: 55693
2024-12-14 22:57:45 - INFO -         Computing Clustering...
2024-12-14 22:58:26 - INFO -             Mean: 0.22748451682862603
2024-12-14 22:58:26 - INFO -               Sd: 0.36689275468965843
2024-12-14 22:58:26 - INFO -              Min: 0.0
2024-12-14 22:58:26 - INFO -               Q1: 0.0
2024-12-14 22:58:26 - INFO -               Q2: 0.0
2024-12-14 22:58:26 - INFO -               Q3: 0.3333333333333333
2024-12-14 22:58:26 - INFO -              Max: 1.0
2024-12-14 22:58:26 - INFO -         Computing Triangles...
2024-12-14 22:58:33 - INFO -             Mean: 5.449733396262477
2024-12-14 22:58:33 - INFO -               Sd: 67.55128348271394
2024-12-14 22:58:33 - INFO -              Min: 0
2024-12-14 22:58:33 - INFO -               Q1: 0.0
2024-12-14 22:58:33 - INFO -               Q2: 0.0
2024-12-14 22:58:33 - INFO -               Q3: 1.0
2024-12-14 22:58:33 - INFO -              Max: 11003
2024-12-14 22:58:33 - INFO -         Computing PageRank...
2024-12-14 22:58:41 - INFO -         Computing Accumulated PageRank...
2024-12-14 22:59:43 - INFO -         Correlations: 
2024-12-14 22:59:43 - INFO -             Popularity and Pagerank: 0.1789509894864646 with p-value of 0.000.
2024-12-14 22:59:43 - INFO -             Popularity and Pagerank CumSum: 0.2909698476642475 with p-value of 0.000
2024-12-14 22:59:43 - INFO -             Popularity and Pagerank Acc: 0.2909698538125426 with p-value of 0.000
2024-12-14 22:59:43 - INFO -             Popularity and Clustering: -0.02920392173063989 with p-value of 0.000
2024-12-14 22:59:43 - INFO -             Popularity and Triangles: 0.16358390142735646 with p-value of 0.000
2024-12-14 22:59:43 - INFO -     Getting 80% of data.
2024-12-14 22:59:43 - INFO -     From 20 percentile to 100 percentile.
2024-12-14 22:59:43 - INFO -         Found: 410197 nodes
2024-12-14 22:59:43 - INFO -         Computing number of Connected Components...
2024-12-14 22:59:46 - INFO -             Number of Connected Components: 54469
2024-12-14 22:59:46 - INFO -         Computing Clustering...
2024-12-14 23:00:21 - INFO -             Mean: 0.1926190443220519
2024-12-14 23:00:21 - INFO -               Sd: 0.3371894193225655
2024-12-14 23:00:21 - INFO -              Min: 0.0
2024-12-14 23:00:21 - INFO -               Q1: 0.0
2024-12-14 23:00:21 - INFO -               Q2: 0.0
2024-12-14 23:00:21 - INFO -               Q3: 0.21794871794871795
2024-12-14 23:00:21 - INFO -              Max: 1.0
2024-12-14 23:00:21 - INFO -         Computing Triangles...
2024-12-14 23:00:26 - INFO -             Mean: 5.979149043020793
2024-12-14 23:00:26 - INFO -               Sd: 73.02036952171848
2024-12-14 23:00:26 - INFO -              Min: 0
2024-12-14 23:00:26 - INFO -               Q1: 0.0
2024-12-14 23:00:26 - INFO -               Q2: 0.0
2024-12-14 23:00:26 - INFO -               Q3: 1.0
2024-12-14 23:00:26 - INFO -              Max: 10744
2024-12-14 23:00:26 - INFO -         Computing PageRank...
2024-12-14 23:00:33 - INFO -         Computing Accumulated PageRank...
2024-12-14 23:01:23 - INFO -         Correlations: 
2024-12-14 23:01:23 - INFO -             Popularity and Pagerank: 0.18208006448471364 with p-value of 0.000.
2024-12-14 23:01:23 - INFO -             Popularity and Pagerank CumSum: 0.3188585581712919 with p-value of 0.000
2024-12-14 23:01:23 - INFO -             Popularity and Pagerank Acc: 0.31885856473172847 with p-value of 0.000
2024-12-14 23:01:23 - INFO -             Popularity and Clustering: -0.024130235428312004 with p-value of 0.000
2024-12-14 23:01:23 - INFO -             Popularity and Triangles: 0.165006975326348 with p-value of 0.000
2024-12-14 23:01:23 - INFO -     Getting 50% of data.
2024-12-14 23:01:23 - INFO -     From 50 percentile to 100 percentile.
2024-12-14 23:01:23 - INFO -         Found: 256373 nodes
2024-12-14 23:01:23 - INFO -         Computing number of Connected Components...
2024-12-14 23:01:25 - INFO -             Number of Connected Components: 45109
2024-12-14 23:01:25 - INFO -         Computing Clustering...
2024-12-14 23:01:50 - INFO -             Mean: 0.15130813558396275
2024-12-14 23:01:50 - INFO -               Sd: 0.29424934885287723
2024-12-14 23:01:50 - INFO -              Min: 0.0
2024-12-14 23:01:50 - INFO -               Q1: 0.0
2024-12-14 23:01:50 - INFO -               Q2: 0.0
2024-12-14 23:01:50 - INFO -               Q3: 0.15384615384615385
2024-12-14 23:01:50 - INFO -              Max: 1.0
2024-12-14 23:01:50 - INFO -         Computing Triangles...
2024-12-14 23:01:54 - INFO -             Mean: 7.3300035495157445
2024-12-14 23:01:54 - INFO -               Sd: 84.78100161216182
2024-12-14 23:01:54 - INFO -              Min: 0
2024-12-14 23:01:54 - INFO -               Q1: 0.0
2024-12-14 23:01:54 - INFO -               Q2: 0.0
2024-12-14 23:01:54 - INFO -               Q3: 1.0
2024-12-14 23:01:54 - INFO -              Max: 9876
2024-12-14 23:01:54 - INFO -         Computing PageRank...
2024-12-14 23:01:58 - INFO -         Computing Accumulated PageRank...
2024-12-14 23:02:30 - INFO -         Correlations: 
2024-12-14 23:02:30 - INFO -             Popularity and Pagerank: 0.19054209701520797 with p-value of 0.000.
2024-12-14 23:02:30 - INFO -             Popularity and Pagerank CumSum: 0.378975962941151 with p-value of 0.000
2024-12-14 23:02:30 - INFO -             Popularity and Pagerank Acc: 0.37897597222543733 with p-value of 0.000
2024-12-14 23:02:30 - INFO -             Popularity and Clustering: -0.019815438847191877 with p-value of 0.000
2024-12-14 23:02:30 - INFO -             Popularity and Triangles: 0.16734737402346478 with p-value of 0.000
2024-12-14 23:02:30 - INFO -     Getting 25% of data.
2024-12-14 23:02:30 - INFO -     From 75 percentile to 100 percentile.
2024-12-14 23:02:30 - INFO -         Found: 128187 nodes
2024-12-14 23:02:30 - INFO -         Computing number of Connected Components...
2024-12-14 23:02:31 - INFO -             Number of Connected Components: 30672
2024-12-14 23:02:31 - INFO -         Computing Clustering...
2024-12-14 23:02:46 - INFO -             Mean: 0.11659603936485283
2024-12-14 23:02:46 - INFO -               Sd: 0.2500445775537542
2024-12-14 23:02:46 - INFO -              Min: 0.0
2024-12-14 23:02:46 - INFO -               Q1: 0.0
2024-12-14 23:02:46 - INFO -               Q2: 0.0
2024-12-14 23:02:46 - INFO -               Q3: 0.10606060606060606
2024-12-14 23:02:46 - INFO -              Max: 1.0
2024-12-14 23:02:46 - INFO -         Computing Triangles...
2024-12-14 23:02:47 - INFO -             Mean: 9.59839921364881
2024-12-14 23:02:47 - INFO -               Sd: 100.4650299177356
2024-12-14 23:02:47 - INFO -              Min: 0
2024-12-14 23:02:47 - INFO -               Q1: 0.0
2024-12-14 23:02:47 - INFO -               Q2: 0.0
2024-12-14 23:02:47 - INFO -               Q3: 1.0
2024-12-14 23:02:47 - INFO -              Max: 8141
2024-12-14 23:02:47 - INFO -         Computing PageRank...
2024-12-14 23:02:49 - INFO -         Computing Accumulated PageRank...
2024-12-14 23:03:05 - INFO -         Correlations: 
2024-12-14 23:03:05 - INFO -             Popularity and Pagerank: 0.1947504953710354 with p-value of 0.000.
2024-12-14 23:03:05 - INFO -             Popularity and Pagerank CumSum: 0.4736696564188445 with p-value of 0.000
2024-12-14 23:03:05 - INFO -             Popularity and Pagerank Acc: 0.4736696784746939 with p-value of 0.000
2024-12-14 23:03:05 - INFO -             Popularity and Clustering: -0.015609065846365074 with p-value of 0.000
2024-12-14 23:03:05 - INFO -             Popularity and Triangles: 0.1710302764538892 with p-value of 0.000
2024-12-14 23:03:05 - INFO -     Getting 10% of data.
2024-12-14 23:03:05 - INFO -     From 90 percentile to 100 percentile.
2024-12-14 23:03:05 - INFO -         Found: 51275 nodes
2024-12-14 23:03:05 - INFO -         Computing number of Connected Components...
2024-12-14 23:03:05 - INFO -             Number of Connected Components: 16501
2024-12-14 23:03:05 - INFO -         Computing Clustering...
2024-12-14 23:03:13 - INFO -             Mean: 0.09205575471289909
2024-12-14 23:03:13 - INFO -               Sd: 0.2114330553768492
2024-12-14 23:03:13 - INFO -              Min: 0.0
2024-12-14 23:03:13 - INFO -               Q1: 0.0
2024-12-14 23:03:13 - INFO -               Q2: 0.0
2024-12-14 23:03:13 - INFO -               Q3: 0.08888888888888889
2024-12-14 23:03:13 - INFO -              Max: 1.0
2024-12-14 23:03:13 - INFO -         Computing Triangles...
2024-12-14 23:03:14 - INFO -             Mean: 14.315163334958557
2024-12-14 23:03:14 - INFO -               Sd: 134.25499140797933
2024-12-14 23:03:14 - INFO -              Min: 0
2024-12-14 23:03:14 - INFO -               Q1: 0.0
2024-12-14 23:03:14 - INFO -               Q2: 0.0
2024-12-14 23:03:14 - INFO -               Q3: 1.0
2024-12-14 23:03:14 - INFO -              Max: 6919
2024-12-14 23:03:14 - INFO -         Computing PageRank...
2024-12-14 23:03:15 - INFO -         Computing Accumulated PageRank...
2024-12-14 23:03:21 - INFO -         Correlations: 
2024-12-14 23:03:21 - INFO -             Popularity and Pagerank: 0.20596590725184716 with p-value of 0.000.
2024-12-14 23:03:21 - INFO -             Popularity and Pagerank CumSum: 0.5931906878005915 with p-value of 0.000
2024-12-14 23:03:21 - INFO -             Popularity and Pagerank Acc: 0.5931907741262814 with p-value of 0.000
2024-12-14 23:03:21 - INFO -             Popularity and Clustering: -0.010689635969383472 with p-value of 0.015
2024-12-14 23:03:21 - INFO -             Popularity and Triangles: 0.1683939687142701 with p-value of 0.000
2024-12-14 23:03:21 - INFO -     Getting 5% of data.
2024-12-14 23:03:21 - INFO -     From 95 percentile to 100 percentile.
2024-12-14 23:03:21 - INFO -         Found: 25638 nodes
2024-12-14 23:03:21 - INFO -         Computing number of Connected Components...
2024-12-14 23:03:22 - INFO -             Number of Connected Components: 9418
2024-12-14 23:03:22 - INFO -         Computing Clustering...
2024-12-14 23:03:26 - INFO -             Mean: 0.08294660110796274
2024-12-14 23:03:26 - INFO -               Sd: 0.19206886325912456
2024-12-14 23:03:26 - INFO -              Min: 0.0
2024-12-14 23:03:26 - INFO -               Q1: 0.0
2024-12-14 23:03:26 - INFO -               Q2: 0.0
2024-12-14 23:03:26 - INFO -               Q3: 0.08421052631578947
2024-12-14 23:03:26 - INFO -              Max: 1.0
2024-12-14 23:03:26 - INFO -         Computing Triangles...
2024-12-14 23:03:27 - INFO -             Mean: 19.400655277322723
2024-12-14 23:03:27 - INFO -               Sd: 162.80469149566798
2024-12-14 23:03:27 - INFO -              Min: 0
2024-12-14 23:03:27 - INFO -               Q1: 0.0
2024-12-14 23:03:27 - INFO -               Q2: 0.0
2024-12-14 23:03:27 - INFO -               Q3: 1.0
2024-12-14 23:03:27 - INFO -              Max: 6396
2024-12-14 23:03:27 - INFO -         Computing PageRank...
2024-12-14 23:03:27 - INFO -         Computing Accumulated PageRank...
2024-12-14 23:03:30 - INFO -         Correlations: 
2024-12-14 23:03:30 - INFO -             Popularity and Pagerank: 0.20622980501331106 with p-value of 0.000.
2024-12-14 23:03:30 - INFO -             Popularity and Pagerank CumSum: 0.6826799920932143 with p-value of 0.000
2024-12-14 23:03:30 - INFO -             Popularity and Pagerank Acc: 0.6826806212399117 with p-value of 0.000
2024-12-14 23:03:30 - INFO -             Popularity and Clustering: -0.010551858166825249 with p-value of 0.091
2024-12-14 23:03:30 - INFO -             Popularity and Triangles: 0.16295732474806265 with p-value of 0.000
2024-12-14 23:03:30 - INFO -     Getting 1% of data.
2024-12-14 23:03:30 - INFO -     From 99 percentile to 100 percentile.
2024-12-14 23:03:30 - INFO -         Found: 5128 nodes
2024-12-14 23:03:30 - INFO -         Computing number of Connected Components...
2024-12-14 23:03:30 - INFO -             Number of Connected Components: 2337
2024-12-14 23:03:30 - INFO -         Computing Clustering...
2024-12-14 23:03:32 - INFO -             Mean: 0.08294550975302079
2024-12-14 23:03:32 - INFO -               Sd: 0.18216002432532133
2024-12-14 23:03:32 - INFO -              Min: 0.0
2024-12-14 23:03:32 - INFO -               Q1: 0.0
2024-12-14 23:03:32 - INFO -               Q2: 0.0
2024-12-14 23:03:32 - INFO -               Q3: 0.09487807219662059
2024-12-14 23:03:32 - INFO -              Max: 1.0
2024-12-14 23:03:32 - INFO -         Computing Triangles...
2024-12-14 23:03:32 - INFO -             Mean: 35.076248049922
2024-12-14 23:03:32 - INFO -               Sd: 202.3915993957045
2024-12-14 23:03:32 - INFO -              Min: 0
2024-12-14 23:03:32 - INFO -               Q1: 0.0
2024-12-14 23:03:32 - INFO -               Q2: 0.0
2024-12-14 23:03:32 - INFO -               Q3: 2.0
2024-12-14 23:03:32 - INFO -              Max: 3991
2024-12-14 23:03:32 - INFO -         Computing PageRank...
2024-12-14 23:03:32 - INFO -         Computing Accumulated PageRank...
2024-12-14 23:03:32 - INFO -         Correlations: 
2024-12-14 23:03:32 - INFO -             Popularity and Pagerank: 0.17840308795260804 with p-value of 0.000.
2024-12-14 23:03:32 - INFO -             Popularity and Pagerank CumSum: 0.8545487623944912 with p-value of 0.000
2024-12-14 23:03:32 - INFO -             Popularity and Pagerank Acc: 0.8545519286985351 with p-value of 0.000
2024-12-14 23:03:32 - INFO -             Popularity and Clustering: -0.013206190175477176 with p-value of 0.344
2024-12-14 23:03:32 - INFO -             Popularity and Triangles: 0.12429730952968057 with p-value of 0.000
2024-12-14 23:03:32 - INFO -     Getting 0.5% of data.
2024-12-14 23:03:32 - INFO -     From 99.5 percentile to 100 percentile.
2024-12-14 23:03:32 - INFO -         Found: 2564 nodes
2024-12-14 23:03:32 - INFO -         Computing number of Connected Components...
2024-12-14 23:03:32 - INFO -             Number of Connected Components: 1254
2024-12-14 23:03:32 - INFO -         Computing Clustering...
2024-12-14 23:03:33 - INFO -             Mean: 0.09008174241257104
2024-12-14 23:03:33 - INFO -               Sd: 0.18698192521404527
2024-12-14 23:03:33 - INFO -              Min: 0.0
2024-12-14 23:03:33 - INFO -               Q1: 0.0
2024-12-14 23:03:33 - INFO -               Q2: 0.0
2024-12-14 23:03:33 - INFO -               Q3: 0.10639980576353852
2024-12-14 23:03:33 - INFO -              Max: 1.0
2024-12-14 23:03:33 - INFO -         Computing Triangles...
2024-12-14 23:03:33 - INFO -             Mean: 39.582683307332296
2024-12-14 23:03:33 - INFO -               Sd: 189.12639116271654
2024-12-14 23:03:33 - INFO -              Min: 0
2024-12-14 23:03:33 - INFO -               Q1: 0.0
2024-12-14 23:03:33 - INFO -               Q2: 0.0
2024-12-14 23:03:33 - INFO -               Q3: 2.0
2024-12-14 23:03:33 - INFO -              Max: 2684
2024-12-14 23:03:33 - INFO -         Computing PageRank...
2024-12-14 23:03:33 - INFO -         Computing Accumulated PageRank...
2024-12-14 23:03:33 - INFO -         Correlations: 
2024-12-14 23:03:33 - INFO -             Popularity and Pagerank: 0.1519681095637988 with p-value of 0.000.
2024-12-14 23:03:33 - INFO -             Popularity and Pagerank CumSum: 0.8871432010644137 with p-value of 0.000
2024-12-14 23:03:33 - INFO -             Popularity and Pagerank Acc: 0.887134900847498 with p-value of 0.000
2024-12-14 23:03:33 - INFO -             Popularity and Clustering: -0.025006641594208963 with p-value of 0.206
2024-12-14 23:03:33 - INFO -             Popularity and Triangles: 0.08595350944928978 with p-value of 0.000
2024-12-14 23:03:33 - INFO -     Getting 0.09999999999999432% of data.
2024-12-14 23:03:33 - INFO -     From 99.9 percentile to 100 percentile.
2024-12-14 23:03:33 - INFO -         Found: 513 nodes
2024-12-14 23:03:33 - INFO -         Computing number of Connected Components...
2024-12-14 23:03:33 - INFO -             Number of Connected Components: 296
2024-12-14 23:03:33 - INFO -         Computing Clustering...
2024-12-14 23:03:33 - INFO -             Mean: 0.0786564482893083
2024-12-14 23:03:33 - INFO -               Sd: 0.1737615175739555
2024-12-14 23:03:33 - INFO -              Min: 0.0
2024-12-14 23:03:33 - INFO -               Q1: 0.0
2024-12-14 23:03:33 - INFO -               Q2: 0.0
2024-12-14 23:03:33 - INFO -               Q3: 0.0
2024-12-14 23:03:33 - INFO -              Max: 1.0
2024-12-14 23:03:33 - INFO -         Computing Triangles...
2024-12-14 23:03:33 - INFO -             Mean: 11.380116959064328
2024-12-14 23:03:33 - INFO -               Sd: 39.860177427070546
2024-12-14 23:03:33 - INFO -              Min: 0
2024-12-14 23:03:33 - INFO -               Q1: 0.0
2024-12-14 23:03:33 - INFO -               Q2: 0.0
2024-12-14 23:03:33 - INFO -               Q3: 0.0
2024-12-14 23:03:33 - INFO -              Max: 381
2024-12-14 23:03:33 - INFO -         Computing PageRank...
2024-12-14 23:03:34 - INFO -         Computing Accumulated PageRank...
2024-12-14 23:03:34 - INFO -         Correlations: 
2024-12-14 23:03:34 - INFO -             Popularity and Pagerank: 0.11171684925299487 with p-value of 0.011.
2024-12-14 23:03:34 - INFO -             Popularity and Pagerank CumSum: 0.9081292697620671 with p-value of 0.000
2024-12-14 23:03:34 - INFO -             Popularity and Pagerank Acc: 0.9085811829703404 with p-value of 0.000
2024-12-14 23:03:34 - INFO -             Popularity and Clustering: 0.0330707302955878 with p-value of 0.455
2024-12-14 23:03:34 - INFO -             Popularity and Triangles: 0.09464072493632525 with p-value of 0.032
2024-12-14 23:03:34 - INFO -     Getting 5% of data.
2024-12-14 23:03:34 - INFO -     From 0 percentile to 5 percentile.
2024-12-14 23:03:34 - INFO -         Found: 25638 nodes
2024-12-14 23:03:34 - INFO -         Computing number of Connected Components...
2024-12-14 23:03:34 - INFO -             Number of Connected Components: 24451
2024-12-14 23:03:34 - INFO -         Computing Clustering...
2024-12-14 23:03:34 - INFO -             Mean: 0.005343646841054061
2024-12-14 23:03:34 - INFO -               Sd: 0.07128949072486265
2024-12-14 23:03:34 - INFO -              Min: 0.0
2024-12-14 23:03:34 - INFO -               Q1: 0.0
2024-12-14 23:03:34 - INFO -               Q2: 0.0
2024-12-14 23:03:34 - INFO -               Q3: 0.0
2024-12-14 23:03:34 - INFO -              Max: 1.0
2024-12-14 23:03:34 - INFO -         Computing Triangles...
2024-12-14 23:03:34 - INFO -             Mean: 0.01298853264685233
2024-12-14 23:03:34 - INFO -               Sd: 0.2833741787315268
2024-12-14 23:03:34 - INFO -              Min: 0
2024-12-14 23:03:34 - INFO -               Q1: 0.0
2024-12-14 23:03:34 - INFO -               Q2: 0.0
2024-12-14 23:03:34 - INFO -               Q3: 0.0
2024-12-14 23:03:34 - INFO -              Max: 16
2024-12-14 23:03:34 - INFO -         Computing PageRank...
2024-12-14 23:03:34 - INFO -         Computing Accumulated PageRank...
2024-12-14 23:03:37 - INFO -         Correlations: 
2024-12-14 23:03:37 - INFO -             Popularity and Pagerank: -0.0587588044998251 with p-value of 0.000.
2024-12-14 23:03:37 - INFO -             Popularity and Pagerank CumSum: 0.6919917299493832 with p-value of 0.000
2024-12-14 23:03:37 - INFO -             Popularity and Pagerank Acc: 0.6921041731401769 with p-value of 0.000
2024-12-14 23:03:37 - INFO -             Popularity and Clustering: -0.04580038437817935 with p-value of 0.000
2024-12-14 23:03:37 - INFO -             Popularity and Triangles: -0.05588682443126271 with p-value of 0.000
2024-12-14 23:03:37 - INFO -     Getting 5% of data.
2024-12-14 23:03:37 - INFO -     From 5 percentile to 10 percentile.
2024-12-14 23:03:37 - INFO -         Found: 25638 nodes
2024-12-14 23:03:37 - INFO -         Computing number of Connected Components...
2024-12-14 23:03:37 - INFO -             Number of Connected Components: 24528
2024-12-14 23:03:37 - INFO -         Computing Clustering...
2024-12-14 23:03:37 - INFO -             Mean: 0.0030748628338143898
2024-12-14 23:03:37 - INFO -               Sd: 0.05468789522793228
2024-12-14 23:03:37 - INFO -              Min: 0.0
2024-12-14 23:03:37 - INFO -               Q1: 0.0
2024-12-14 23:03:37 - INFO -               Q2: 0.0
2024-12-14 23:03:37 - INFO -               Q3: 0.0
2024-12-14 23:03:37 - INFO -              Max: 1.0
2024-12-14 23:03:37 - INFO -         Computing Triangles...
2024-12-14 23:03:37 - INFO -             Mean: 0.003627428036508308
2024-12-14 23:03:37 - INFO -               Sd: 0.06745782205259292
2024-12-14 23:03:37 - INFO -              Min: 0
2024-12-14 23:03:37 - INFO -               Q1: 0.0
2024-12-14 23:03:37 - INFO -               Q2: 0.0
2024-12-14 23:03:37 - INFO -               Q3: 0.0
2024-12-14 23:03:37 - INFO -              Max: 3
2024-12-14 23:03:37 - INFO -         Computing PageRank...
2024-12-14 23:03:37 - INFO -         Computing Accumulated PageRank...
2024-12-14 23:03:40 - INFO -         Correlations: 
2024-12-14 23:03:40 - INFO -             Popularity and Pagerank: -0.0038475535277357788 with p-value of 0.538.
2024-12-14 23:03:40 - INFO -             Popularity and Pagerank CumSum: 0.9283206594517329 with p-value of 0.000
2024-12-14 23:03:40 - INFO -             Popularity and Pagerank Acc: 0.928353179789407 with p-value of 0.000
2024-12-14 23:03:40 - INFO -             Popularity and Clustering: -0.004863897703512706 with p-value of 0.436
2024-12-14 23:03:40 - INFO -             Popularity and Triangles: -0.003995736205762962 with p-value of 0.522
2024-12-14 23:03:40 - INFO -     Getting 5% of data.
2024-12-14 23:03:40 - INFO -     From 10 percentile to 15 percentile.
2024-12-14 23:03:40 - INFO -         Found: 25638 nodes
2024-12-14 23:03:40 - INFO -         Computing number of Connected Components...
2024-12-14 23:03:40 - INFO -             Number of Connected Components: 24467
2024-12-14 23:03:40 - INFO -         Computing Clustering...
2024-12-14 23:03:40 - INFO -             Mean: 0.003988653994504685
2024-12-14 23:03:40 - INFO -               Sd: 0.06130929303208153
2024-12-14 23:03:40 - INFO -              Min: 0.0
2024-12-14 23:03:40 - INFO -               Q1: 0.0
2024-12-14 23:03:40 - INFO -               Q2: 0.0
2024-12-14 23:03:40 - INFO -               Q3: 0.0
2024-12-14 23:03:40 - INFO -              Max: 1.0
2024-12-14 23:03:40 - INFO -         Computing Triangles...
2024-12-14 23:03:41 - INFO -             Mean: 0.005148607535689211
2024-12-14 23:03:41 - INFO -               Sd: 0.07783602086727552
2024-12-14 23:03:41 - INFO -              Min: 0
2024-12-14 23:03:41 - INFO -               Q1: 0.0
2024-12-14 23:03:41 - INFO -               Q2: 0.0
2024-12-14 23:03:41 - INFO -               Q3: 0.0
2024-12-14 23:03:41 - INFO -              Max: 3
2024-12-14 23:03:41 - INFO -         Computing PageRank...
2024-12-14 23:03:41 - INFO -         Computing Accumulated PageRank...
2024-12-14 23:03:44 - INFO -         Correlations: 
2024-12-14 23:03:44 - INFO -             Popularity and Pagerank: 0.004188055461242431 with p-value of 0.503.
2024-12-14 23:03:44 - INFO -             Popularity and Pagerank CumSum: 0.9876850742963177 with p-value of 0.000
2024-12-14 23:03:44 - INFO -             Popularity and Pagerank Acc: 0.9876917388497967 with p-value of 0.000
2024-12-14 23:03:44 - INFO -             Popularity and Clustering: 0.010621808974997888 with p-value of 0.089
2024-12-14 23:03:44 - INFO -             Popularity and Triangles: 0.01017599889060212 with p-value of 0.103
2024-12-14 23:03:44 - INFO -     Getting 10% of data.
2024-12-14 23:03:44 - INFO -     From 20 percentile to 30 percentile.
2024-12-14 23:03:44 - INFO -         Found: 51275 nodes
2024-12-14 23:03:44 - INFO -         Computing number of Connected Components...
2024-12-14 23:03:44 - INFO -             Number of Connected Components: 46292
2024-12-14 23:03:44 - INFO -         Computing Clustering...
2024-12-14 23:03:44 - INFO -             Mean: 0.010311928045228874
2024-12-14 23:03:44 - INFO -               Sd: 0.09753951868387895
2024-12-14 23:03:44 - INFO -              Min: 0.0
2024-12-14 23:03:44 - INFO -               Q1: 0.0
2024-12-14 23:03:44 - INFO -               Q2: 0.0
2024-12-14 23:03:44 - INFO -               Q3: 0.0
2024-12-14 23:03:44 - INFO -              Max: 1.0
2024-12-14 23:03:44 - INFO -         Computing Triangles...
2024-12-14 23:03:44 - INFO -             Mean: 0.017318381277425646
2024-12-14 23:03:44 - INFO -               Sd: 0.1904936011951562
2024-12-14 23:03:44 - INFO -              Min: 0
2024-12-14 23:03:44 - INFO -               Q1: 0.0
2024-12-14 23:03:44 - INFO -               Q2: 0.0
2024-12-14 23:03:44 - INFO -               Q3: 0.0
2024-12-14 23:03:44 - INFO -              Max: 9
2024-12-14 23:03:44 - INFO -         Computing PageRank...
2024-12-14 23:03:45 - INFO -         Computing Accumulated PageRank...
2024-12-14 23:03:51 - INFO -         Correlations: 
2024-12-14 23:03:51 - INFO -             Popularity and Pagerank: 0.007379124700519642 with p-value of 0.095.
2024-12-14 23:03:51 - INFO -             Popularity and Pagerank CumSum: 0.992521037480679 with p-value of 0.000
2024-12-14 23:03:51 - INFO -             Popularity and Pagerank Acc: 0.9925213517468902 with p-value of 0.000
2024-12-14 23:03:51 - INFO -             Popularity and Clustering: -0.0074927965445457 with p-value of 0.090
2024-12-14 23:03:51 - INFO -             Popularity and Triangles: 0.0012717398745582605 with p-value of 0.773
2024-12-14 23:03:51 - INFO -     Getting 10% of data.
2024-12-14 23:03:51 - INFO -     From 35 percentile to 45 percentile.
2024-12-14 23:03:51 - INFO -         Found: 51275 nodes
2024-12-14 23:03:51 - INFO -         Computing number of Connected Components...
2024-12-14 23:03:51 - INFO -             Number of Connected Components: 45240
2024-12-14 23:03:51 - INFO -         Computing Clustering...
2024-12-14 23:03:51 - INFO -             Mean: 0.010613804311512745
2024-12-14 23:03:51 - INFO -               Sd: 0.09757710938396344
2024-12-14 23:03:51 - INFO -              Min: 0.0
2024-12-14 23:03:51 - INFO -               Q1: 0.0
2024-12-14 23:03:51 - INFO -               Q2: 0.0
2024-12-14 23:03:51 - INFO -               Q3: 0.0
2024-12-14 23:03:51 - INFO -              Max: 1.0
2024-12-14 23:03:51 - INFO -         Computing Triangles...
2024-12-14 23:03:51 - INFO -             Mean: 0.01971721111652852
2024-12-14 23:03:51 - INFO -               Sd: 0.2022355216494353
2024-12-14 23:03:51 - INFO -              Min: 0
2024-12-14 23:03:51 - INFO -               Q1: 0.0
2024-12-14 23:03:51 - INFO -               Q2: 0.0
2024-12-14 23:03:51 - INFO -               Q3: 0.0
2024-12-14 23:03:51 - INFO -              Max: 11
2024-12-14 23:03:51 - INFO -         Computing PageRank...
2024-12-14 23:03:52 - INFO -         Computing Accumulated PageRank...
2024-12-14 23:03:57 - INFO -         Correlations: 
2024-12-14 23:03:57 - INFO -             Popularity and Pagerank: 0.0058072847739965225 with p-value of 0.189.
2024-12-14 23:03:57 - INFO -             Popularity and Pagerank CumSum: 0.9937776578974302 with p-value of 0.000
2024-12-14 23:03:57 - INFO -             Popularity and Pagerank Acc: 0.9937778198710991 with p-value of 0.000
2024-12-14 23:03:57 - INFO -             Popularity and Clustering: -0.004290569586653516 with p-value of 0.331
2024-12-14 23:03:57 - INFO -             Popularity and Triangles: 0.0006724619584568595 with p-value of 0.879
2024-12-14 23:03:57 - INFO -     Getting 10% of data.
2024-12-14 23:03:57 - INFO -     From 60 percentile to 70 percentile.
2024-12-14 23:03:58 - INFO -         Found: 51276 nodes
2024-12-14 23:03:58 - INFO -         Computing number of Connected Components...
2024-12-14 23:03:58 - INFO -             Number of Connected Components: 40906
2024-12-14 23:03:58 - INFO -         Computing Clustering...
2024-12-14 23:03:58 - INFO -             Mean: 0.017508124438877053
2024-12-14 23:03:58 - INFO -               Sd: 0.11958067357910901
2024-12-14 23:03:58 - INFO -              Min: 0.0
2024-12-14 23:03:58 - INFO -               Q1: 0.0
2024-12-14 23:03:58 - INFO -               Q2: 0.0
2024-12-14 23:03:58 - INFO -               Q3: 0.0
2024-12-14 23:03:58 - INFO -              Max: 1.0
2024-12-14 23:03:58 - INFO -         Computing Triangles...
2024-12-14 23:03:58 - INFO -             Mean: 0.050257430376784464
2024-12-14 23:03:58 - INFO -               Sd: 0.43128478368713524
2024-12-14 23:03:58 - INFO -              Min: 0
2024-12-14 23:03:58 - INFO -               Q1: 0.0
2024-12-14 23:03:58 - INFO -               Q2: 0.0
2024-12-14 23:03:58 - INFO -               Q3: 0.0
2024-12-14 23:03:58 - INFO -              Max: 24
2024-12-14 23:03:58 - INFO -         Computing PageRank...
2024-12-14 23:03:59 - INFO -         Computing Accumulated PageRank...
2024-12-14 23:04:05 - INFO -         Correlations: 
2024-12-14 23:04:05 - INFO -             Popularity and Pagerank: 0.025348873767086756 with p-value of 0.000.
2024-12-14 23:04:05 - INFO -             Popularity and Pagerank CumSum: 0.9923438642225451 with p-value of 0.000
2024-12-14 23:04:05 - INFO -             Popularity and Pagerank Acc: 0.9923438698103684 with p-value of 0.000
2024-12-14 23:04:05 - INFO -             Popularity and Clustering: 0.002055960855841381 with p-value of 0.642
2024-12-14 23:04:05 - INFO -             Popularity and Triangles: 0.01167260008696389 with p-value of 0.008
2024-12-14 23:04:05 - INFO -     Getting 10% of data.
2024-12-14 23:04:05 - INFO -     From 80 percentile to 90 percentile.
2024-12-14 23:04:05 - INFO -         Found: 51276 nodes
2024-12-14 23:04:05 - INFO -         Computing number of Connected Components...
2024-12-14 23:04:05 - INFO -             Number of Connected Components: 29599
2024-12-14 23:04:05 - INFO -         Computing Clustering...
2024-12-14 23:04:06 - INFO -             Mean: 0.04093640350128101
2024-12-14 23:04:06 - INFO -               Sd: 0.16399580987970847
2024-12-14 23:04:06 - INFO -              Min: 0.0
2024-12-14 23:04:06 - INFO -               Q1: 0.0
2024-12-14 23:04:06 - INFO -               Q2: 0.0
2024-12-14 23:04:06 - INFO -               Q3: 0.0
2024-12-14 23:04:06 - INFO -              Max: 1.0
2024-12-14 23:04:06 - INFO -         Computing Triangles...
2024-12-14 23:04:07 - INFO -             Mean: 0.3940439971916686
2024-12-14 23:04:07 - INFO -               Sd: 2.509027449589504
2024-12-14 23:04:07 - INFO -              Min: 0
2024-12-14 23:04:07 - INFO -               Q1: 0.0
2024-12-14 23:04:07 - INFO -               Q2: 0.0
2024-12-14 23:04:07 - INFO -               Q3: 0.0
2024-12-14 23:04:07 - INFO -              Max: 111
2024-12-14 23:04:07 - INFO -         Computing PageRank...
2024-12-14 23:04:07 - INFO -         Computing Accumulated PageRank...
2024-12-14 23:04:13 - INFO -         Correlations: 
2024-12-14 23:04:13 - INFO -             Popularity and Pagerank: 0.05810397252346 with p-value of 0.000.
2024-12-14 23:04:13 - INFO -             Popularity and Pagerank CumSum: 0.983017301341141 with p-value of 0.000
2024-12-14 23:04:13 - INFO -             Popularity and Pagerank Acc: 0.9830172924303138 with p-value of 0.000
2024-12-14 23:04:13 - INFO -             Popularity and Clustering: -0.0103776962561792 with p-value of 0.019
2024-12-14 23:04:13 - INFO -             Popularity and Triangles: 0.027581146730054424 with p-value of 0.000
2024-12-14 23:04:13 - INFO -     Getting 5% of data.
2024-12-14 23:04:13 - INFO -     From 90 percentile to 95 percentile.
2024-12-14 23:04:13 - INFO -         Found: 25638 nodes
2024-12-14 23:04:13 - INFO -         Computing number of Connected Components...
2024-12-14 23:04:13 - INFO -             Number of Connected Components: 15222
2024-12-14 23:04:13 - INFO -         Computing Clustering...
2024-12-14 23:04:14 - INFO -             Mean: 0.03701365880608369
2024-12-14 23:04:14 - INFO -               Sd: 0.1508241660357777
2024-12-14 23:04:14 - INFO -              Min: 0.0
2024-12-14 23:04:14 - INFO -               Q1: 0.0
2024-12-14 23:04:14 - INFO -               Q2: 0.0
2024-12-14 23:04:14 - INFO -               Q3: 0.0
2024-12-14 23:04:14 - INFO -              Max: 1.0
2024-12-14 23:04:14 - INFO -         Computing Triangles...
2024-12-14 23:04:14 - INFO -             Mean: 0.5682190498478821
2024-12-14 23:04:14 - INFO -               Sd: 4.389481770753206
2024-12-14 23:04:14 - INFO -              Min: 0
2024-12-14 23:04:14 - INFO -               Q1: 0.0
2024-12-14 23:04:14 - INFO -               Q2: 0.0
2024-12-14 23:04:14 - INFO -               Q3: 0.0
2024-12-14 23:04:14 - INFO -              Max: 203
2024-12-14 23:04:14 - INFO -         Computing PageRank...
2024-12-14 23:04:14 - INFO -         Computing Accumulated PageRank...
2024-12-14 23:04:17 - INFO -         Correlations: 
2024-12-14 23:04:17 - INFO -             Popularity and Pagerank: 0.059360543047115875 with p-value of 0.000.
2024-12-14 23:04:17 - INFO -             Popularity and Pagerank CumSum: 0.9888972112126178 with p-value of 0.000
2024-12-14 23:04:17 - INFO -             Popularity and Pagerank Acc: 0.9888973559511233 with p-value of 0.000
2024-12-14 23:04:17 - INFO -             Popularity and Clustering: -0.003855276591035274 with p-value of 0.537
2024-12-14 23:04:17 - INFO -             Popularity and Triangles: 0.02837472283992618 with p-value of 0.000
2024-12-14 23:04:17 - INFO -     Getting 4% of data.
2024-12-14 23:04:17 - INFO -     From 95 percentile to 99 percentile.
2024-12-14 23:04:17 - INFO -         Found: 20511 nodes
2024-12-14 23:04:17 - INFO -         Computing number of Connected Components...
2024-12-14 23:04:17 - INFO -             Number of Connected Components: 9162
2024-12-14 23:04:17 - INFO -         Computing Clustering...
2024-12-14 23:04:19 - INFO -             Mean: 0.061680892523199166
2024-12-14 23:04:19 - INFO -               Sd: 0.17293832034651746
2024-12-14 23:04:19 - INFO -              Min: 0.0
2024-12-14 23:04:19 - INFO -               Q1: 0.0
2024-12-14 23:04:19 - INFO -               Q2: 0.0
2024-12-14 23:04:19 - INFO -               Q3: 0.0
2024-12-14 23:04:19 - INFO -              Max: 1.0
2024-12-14 23:04:19 - INFO -         Computing Triangles...
2024-12-14 23:04:19 - INFO -             Mean: 2.815269855199649
2024-12-14 23:04:19 - INFO -               Sd: 14.844559565128893
2024-12-14 23:04:19 - INFO -              Min: 0
2024-12-14 23:04:19 - INFO -               Q1: 0.0
2024-12-14 23:04:19 - INFO -               Q2: 0.0
2024-12-14 23:04:19 - INFO -               Q3: 0.0
2024-12-14 23:04:19 - INFO -              Max: 460
2024-12-14 23:04:19 - INFO -         Computing PageRank...
2024-12-14 23:04:19 - INFO -         Computing Accumulated PageRank...
2024-12-14 23:04:22 - INFO -         Correlations: 
2024-12-14 23:04:22 - INFO -             Popularity and Pagerank: 0.10716521311152208 with p-value of 0.000.
2024-12-14 23:04:22 - INFO -             Popularity and Pagerank CumSum: 0.9472264970251948 with p-value of 0.000
2024-12-14 23:04:22 - INFO -             Popularity and Pagerank Acc: 0.9472264954802005 with p-value of 0.000
2024-12-14 23:04:22 - INFO -             Popularity and Clustering: -0.009133754944040177 with p-value of 0.191
2024-12-14 23:04:22 - INFO -             Popularity and Triangles: 0.06465023610990306 with p-value of 0.000
2024-12-14 23:04:22 - INFO -     Getting 5% of data.
2024-12-14 23:04:22 - INFO -     From 95 percentile to 100 percentile.
2024-12-14 23:04:22 - INFO -         Found: 25638 nodes
2024-12-14 23:04:22 - INFO -         Computing number of Connected Components...
2024-12-14 23:04:22 - INFO -             Number of Connected Components: 9418
2024-12-14 23:04:22 - INFO -         Computing Clustering...
2024-12-14 23:04:27 - INFO -             Mean: 0.08294660110796274
2024-12-14 23:04:27 - INFO -               Sd: 0.19206886325912456
2024-12-14 23:04:27 - INFO -              Min: 0.0
2024-12-14 23:04:27 - INFO -               Q1: 0.0
2024-12-14 23:04:27 - INFO -               Q2: 0.0
2024-12-14 23:04:27 - INFO -               Q3: 0.08421052631578947
2024-12-14 23:04:27 - INFO -              Max: 1.0
2024-12-14 23:04:27 - INFO -         Computing Triangles...
2024-12-14 23:04:27 - INFO -             Mean: 19.400655277322723
2024-12-14 23:04:27 - INFO -               Sd: 162.80469149566798
2024-12-14 23:04:27 - INFO -              Min: 0
2024-12-14 23:04:27 - INFO -               Q1: 0.0
2024-12-14 23:04:27 - INFO -               Q2: 0.0
2024-12-14 23:04:27 - INFO -               Q3: 1.0
2024-12-14 23:04:27 - INFO -              Max: 6396
2024-12-14 23:04:27 - INFO -         Computing PageRank...
2024-12-14 23:04:27 - INFO -         Computing Accumulated PageRank...
2024-12-14 23:04:30 - INFO -         Correlations: 
2024-12-14 23:04:30 - INFO -             Popularity and Pagerank: 0.20622980501331106 with p-value of 0.000.
2024-12-14 23:04:30 - INFO -             Popularity and Pagerank CumSum: 0.6826799920932143 with p-value of 0.000
2024-12-14 23:04:30 - INFO -             Popularity and Pagerank Acc: 0.6826806212399117 with p-value of 0.000
2024-12-14 23:04:30 - INFO -             Popularity and Clustering: -0.010551858166825249 with p-value of 0.091
2024-12-14 23:04:30 - INFO -             Popularity and Triangles: 0.16295732474806265 with p-value of 0.000
2024-12-14 23:04:30 - INFO -     Getting 2% of data.
2024-12-14 23:04:30 - INFO -     From 97 percentile to 99 percentile.
2024-12-14 23:04:30 - INFO -         Found: 10256 nodes
2024-12-14 23:04:30 - INFO -         Computing number of Connected Components...
2024-12-14 23:04:31 - INFO -             Number of Connected Components: 5294
2024-12-14 23:04:31 - INFO -         Computing Clustering...
2024-12-14 23:04:31 - INFO -             Mean: 0.05135613856228283
2024-12-14 23:04:31 - INFO -               Sd: 0.15987970107510294
2024-12-14 23:04:31 - INFO -              Min: 0.0
2024-12-14 23:04:31 - INFO -               Q1: 0.0
2024-12-14 23:04:31 - INFO -               Q2: 0.0
2024-12-14 23:04:31 - INFO -               Q3: 0.0
2024-12-14 23:04:31 - INFO -              Max: 1.0
2024-12-14 23:04:31 - INFO -         Computing Triangles...
2024-12-14 23:04:31 - INFO -             Mean: 1.7351794071762872
2024-12-14 23:04:31 - INFO -               Sd: 9.1014047129759
2024-12-14 23:04:31 - INFO -              Min: 0
2024-12-14 23:04:31 - INFO -               Q1: 0.0
2024-12-14 23:04:31 - INFO -               Q2: 0.0
2024-12-14 23:04:31 - INFO -               Q3: 0.0
2024-12-14 23:04:31 - INFO -              Max: 217
2024-12-14 23:04:31 - INFO -         Computing PageRank...
2024-12-14 23:04:31 - INFO -         Computing Accumulated PageRank...
2024-12-14 23:04:32 - INFO -         Correlations: 
2024-12-14 23:04:32 - INFO -             Popularity and Pagerank: 0.09094306474851512 with p-value of 0.000.
2024-12-14 23:04:32 - INFO -             Popularity and Pagerank CumSum: 0.9771174183877898 with p-value of 0.000
2024-12-14 23:04:32 - INFO -             Popularity and Pagerank Acc: 0.9771171400685628 with p-value of 0.000
2024-12-14 23:04:32 - INFO -             Popularity and Clustering: -0.019103253753433573 with p-value of 0.053
2024-12-14 23:04:32 - INFO -             Popularity and Triangles: 0.048051588450829935 with p-value of 0.000
2024-12-14 23:04:32 - INFO -     Getting 0.5% of data.
2024-12-14 23:04:32 - INFO -     From 99 percentile to 99.5 percentile.
2024-12-14 23:04:32 - INFO -         Found: 2565 nodes
2024-12-14 23:04:32 - INFO -         Computing number of Connected Components...
2024-12-14 23:04:32 - INFO -             Number of Connected Components: 1584
2024-12-14 23:04:32 - INFO -         Computing Clustering...
2024-12-14 23:04:33 - INFO -             Mean: 0.04238056473681339
2024-12-14 23:04:33 - INFO -               Sd: 0.1489582078073624
2024-12-14 23:04:33 - INFO -              Min: 0.0
2024-12-14 23:04:33 - INFO -               Q1: 0.0
2024-12-14 23:04:33 - INFO -               Q2: 0.0
2024-12-14 23:04:33 - INFO -               Q3: 0.0
2024-12-14 23:04:33 - INFO -              Max: 1.0
2024-12-14 23:04:33 - INFO -         Computing Triangles...
2024-12-14 23:04:33 - INFO -             Mean: 0.9005847953216374
2024-12-14 23:04:33 - INFO -               Sd: 5.379790037123055
2024-12-14 23:04:33 - INFO -              Min: 0
2024-12-14 23:04:33 - INFO -               Q1: 0.0
2024-12-14 23:04:33 - INFO -               Q2: 0.0
2024-12-14 23:04:33 - INFO -               Q3: 0.0
2024-12-14 23:04:33 - INFO -              Max: 118
2024-12-14 23:04:33 - INFO -         Computing PageRank...
2024-12-14 23:04:33 - INFO -         Computing Accumulated PageRank...
2024-12-14 23:04:33 - INFO -         Correlations: 
2024-12-14 23:04:33 - INFO -             Popularity and Pagerank: 0.04501996406734589 with p-value of 0.023.
2024-12-14 23:04:33 - INFO -             Popularity and Pagerank CumSum: 0.9911563259149729 with p-value of 0.000
2024-12-14 23:04:33 - INFO -             Popularity and Pagerank Acc: 0.9911585419829673 with p-value of 0.000
2024-12-14 23:04:33 - INFO -             Popularity and Clustering: 0.01780405867306022 with p-value of 0.367
2024-12-14 23:04:33 - INFO -             Popularity and Triangles: 0.00675303820375014 with p-value of 0.732
2024-12-14 23:04:33 - INFO -     Getting 0.20000000000000284% of data.
2024-12-14 23:04:33 - INFO -     From 99.5 percentile to 99.7 percentile.
2024-12-14 23:04:33 - INFO -         Found: 1026 nodes
2024-12-14 23:04:33 - INFO -         Computing number of Connected Components...
2024-12-14 23:04:33 - INFO -             Number of Connected Components: 680
2024-12-14 23:04:33 - INFO -         Computing Clustering...
2024-12-14 23:04:33 - INFO -             Mean: 0.04505015666069503
2024-12-14 23:04:33 - INFO -               Sd: 0.15301288441677163
2024-12-14 23:04:33 - INFO -              Min: 0.0
2024-12-14 23:04:33 - INFO -               Q1: 0.0
2024-12-14 23:04:33 - INFO -               Q2: 0.0
2024-12-14 23:04:33 - INFO -               Q3: 0.0
2024-12-14 23:04:33 - INFO -              Max: 1.0
2024-12-14 23:04:33 - INFO -         Computing Triangles...
2024-12-14 23:04:33 - INFO -             Mean: 1.5292397660818713
2024-12-14 23:04:33 - INFO -               Sd: 8.341469883163544
2024-12-14 23:04:33 - INFO -              Min: 0
2024-12-14 23:04:33 - INFO -               Q1: 0.0
2024-12-14 23:04:33 - INFO -               Q2: 0.0
2024-12-14 23:04:33 - INFO -               Q3: 0.0
2024-12-14 23:04:33 - INFO -              Max: 128
2024-12-14 23:04:33 - INFO -         Computing PageRank...
2024-12-14 23:04:33 - INFO -         Computing Accumulated PageRank...
2024-12-14 23:04:33 - INFO -         Correlations: 
2024-12-14 23:04:33 - INFO -             Popularity and Pagerank: 0.0376736149754085 with p-value of 0.228.
2024-12-14 23:04:33 - INFO -             Popularity and Pagerank CumSum: 0.9967418435502826 with p-value of 0.000
2024-12-14 23:04:33 - INFO -             Popularity and Pagerank Acc: 0.9967887793519725 with p-value of 0.000
2024-12-14 23:04:33 - INFO -             Popularity and Clustering: -0.011657840456956012 with p-value of 0.709
2024-12-14 23:04:33 - INFO -             Popularity and Triangles: 0.030226456141357466 with p-value of 0.333
2024-12-14 23:04:33 - INFO -     Getting 0.10000000000000853% of data.
2024-12-14 23:04:33 - INFO -     From 99.8 percentile to 99.9 percentile.
2024-12-14 23:04:33 - INFO -         Found: 514 nodes
2024-12-14 23:04:33 - INFO -         Computing number of Connected Components...
2024-12-14 23:04:33 - INFO -             Number of Connected Components: 323
2024-12-14 23:04:33 - INFO -         Computing Clustering...
2024-12-14 23:04:33 - INFO -             Mean: 0.05394924086101342
2024-12-14 23:04:33 - INFO -               Sd: 0.15741227902355917
2024-12-14 23:04:33 - INFO -              Min: 0.0
2024-12-14 23:04:33 - INFO -               Q1: 0.0
2024-12-14 23:04:33 - INFO -               Q2: 0.0
2024-12-14 23:04:33 - INFO -               Q3: 0.0
2024-12-14 23:04:33 - INFO -              Max: 1.0
2024-12-14 23:04:33 - INFO -         Computing Triangles...
2024-12-14 23:04:33 - INFO -             Mean: 1.8560311284046693
2024-12-14 23:04:33 - INFO -               Sd: 8.423413288556388
2024-12-14 23:04:33 - INFO -              Min: 0
2024-12-14 23:04:33 - INFO -               Q1: 0.0
2024-12-14 23:04:33 - INFO -               Q2: 0.0
2024-12-14 23:04:33 - INFO -               Q3: 0.0
2024-12-14 23:04:33 - INFO -              Max: 80
2024-12-14 23:04:33 - INFO -         Computing PageRank...
2024-12-14 23:04:33 - INFO -         Computing Accumulated PageRank...
2024-12-14 23:04:33 - INFO -         Correlations: 
2024-12-14 23:04:33 - INFO -             Popularity and Pagerank: 0.058312513964566055 with p-value of 0.187.
2024-12-14 23:04:33 - INFO -             Popularity and Pagerank CumSum: 0.9974751047463978 with p-value of 0.000
2024-12-14 23:04:33 - INFO -             Popularity and Pagerank Acc: 0.9975320945880535 with p-value of 0.000
2024-12-14 23:04:33 - INFO -             Popularity and Clustering: 0.022485436443676458 with p-value of 0.611
2024-12-14 23:04:33 - INFO -             Popularity and Triangles: 0.026119201347693938 with p-value of 0.555
2024-12-14 23:04:33 - INFO -     Getting 0.04999999999999716% of data.
2024-12-14 23:04:33 - INFO -     From 99.9 percentile to 99.95 percentile.
2024-12-14 23:04:33 - INFO -         Found: 257 nodes
2024-12-14 23:04:33 - INFO -         Computing number of Connected Components...
2024-12-14 23:04:33 - INFO -             Number of Connected Components: 197
2024-12-14 23:04:33 - INFO -         Computing Clustering...
2024-12-14 23:04:33 - INFO -             Mean: 0.06503142339718214
2024-12-14 23:04:33 - INFO -               Sd: 0.2061388159392726
2024-12-14 23:04:33 - INFO -              Min: 0.0
2024-12-14 23:04:33 - INFO -               Q1: 0.0
2024-12-14 23:04:33 - INFO -               Q2: 0.0
2024-12-14 23:04:33 - INFO -               Q3: 0.0
2024-12-14 23:04:33 - INFO -              Max: 1.0
2024-12-14 23:04:33 - INFO -         Computing Triangles...
2024-12-14 23:04:33 - INFO -             Mean: 1.2957198443579767
2024-12-14 23:04:33 - INFO -               Sd: 5.732829421080504
2024-12-14 23:04:33 - INFO -              Min: 0
2024-12-14 23:04:33 - INFO -               Q1: 0.0
2024-12-14 23:04:33 - INFO -               Q2: 0.0
2024-12-14 23:04:33 - INFO -               Q3: 0.0
2024-12-14 23:04:33 - INFO -              Max: 48
2024-12-14 23:04:33 - INFO -         Computing PageRank...
2024-12-14 23:04:33 - INFO -         Computing Accumulated PageRank...
2024-12-14 23:04:33 - INFO -         Correlations: 
2024-12-14 23:04:33 - INFO -             Popularity and Pagerank: 0.16745426269344713 with p-value of 0.007.
2024-12-14 23:04:33 - INFO -             Popularity and Pagerank CumSum: 0.9854481438714144 with p-value of 0.000
2024-12-14 23:04:33 - INFO -             Popularity and Pagerank Acc: 0.9849301940149133 with p-value of 0.000
2024-12-14 23:04:33 - INFO -             Popularity and Clustering: 0.0687415551298782 with p-value of 0.272
2024-12-14 23:04:33 - INFO -             Popularity and Triangles: 0.10645647557209047 with p-value of 0.089
2024-12-14 23:04:33 - INFO - DONE!
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
