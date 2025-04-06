#!/usr/bin/env python3


import logging
import os
from neo4j import GraphDatabase, basic_auth
import torch
from dotenv import load_dotenv
import pickle
import numpy as np
from tqdm import tqdm


def get_positive_info(all_tags: list[str]) -> tuple[list, list, list, list]:
    with driver.session() as session:
        query = f"""
            MATCH (n:Artist)-[:COLLAB_WITH]->(m:Artist)
            WHERE 1 = 1
            AND n.popularity_scaled >= {perc_value}
            AND m.popularity_scaled >= {perc_value}
            OPTIONAL MATCH (n)-[r0:LAST_FM_MATCH]->(m)
            OPTIONAL MATCH (n)-[r1:MUSICALLY_RELATED_TO]->(m)
            OPTIONAL MATCH (n)-[r2:PERSONALLY_RELATED_TO]->(m)
            OPTIONAL MATCH (n)-[r3:LINKED_TO]->(m)
            WITH
                n,
                m,
                {all_tags} AS all_tags,
                CASE
                  WHEN r0 IS NOT NULL THEN r0.pg_weight
                  ELSE 0
                END AS lfm,
                CASE
                  WHEN r1 IS NOT NULL THEN r1.pg_weight
                  ELSE 0
                END AS mrt,
                CASE
                  WHEN r2 IS NOT NULL THEN r2.pg_weight
                  ELSE 0
                END AS prt,
                CASE
                  WHEN r3 IS NOT NULL THEN r3.pg_weight
                  ELSE 0
                END AS lt
            RETURN
                n.main_id AS id0,
                CASE WHEN n.begin_date IS NULL THEN 0 ELSE 1 END AS hbd0,
                COALESCE(n.begin_date, 0) AS bd0,
                CASE WHEN n.end_date IS NULL THEN 0 else 1 END AS hed0,
                COALESCE(n.end_date, 0) AS ed0,
                n.ended AS e0,
                n.gender_1 AS g10,
                n.gender_2 AS g20,
                n.gender_3 AS g30,
                n.gender_4 AS g40,
                n.gender_5 AS g50,
                n.popularity_scaled AS ps0,
                n.type_1 AS t10,
                n.type_2 AS t20,
                n.type_3 AS t30,
                n.type_4 AS t40,
                n.type_5 AS t50,
                n.type_6 AS t60,
                CASE
                  WHEN size(n.tags) = 0 THEN [tag IN all_tags | 0]
                  ELSE  [tag IN all_tags | toFloat(size([x IN n.tags WHERE x = tag])) / size(n.tags)]
                END AS tags0,
                COALESCE(n.collab_count_{year}_{month}, 0) AS cc0,
                COALESCE(n.collab_popularity_{year}_{month}, 0) AS cp0,
                COALESCE(n.solo_count_{year}_{month}, 0) AS sc0,
                COALESCE(n.solo_popularity_{year}_{month}, 0) AS sp0,

                m.main_id AS id1,
                CASE WHEN m.begin_date IS NULL THEN 0 ELSE 1 END AS hbd1,
                COALESCE(m.begin_date, 0) AS bd1,
                CASE WHEN m.end_date IS NULL THEN 0 else 1 END AS hed1,
                COALESCE(m.end_date, 0) AS ed1,
                m.ended AS e1,
                m.gender_1 AS g11,
                m.gender_2 AS g21,
                m.gender_3 AS g31,
                m.gender_4 AS g41,
                m.gender_5 AS g51,
                m.popularity_scaled AS ps1,
                m.type_1 AS t11,
                m.type_2 AS t21,
                m.type_3 AS t31,
                m.type_4 AS t41,
                m.type_5 AS t51,
                m.type_6 AS t61,
                CASE
                  WHEN size(m.tags) = 0 THEN [tag IN all_tags | 0]
                  ELSE  [tag IN all_tags | toFloat(size([x IN m.tags WHERE x = tag])) / size(m.tags)]
                END AS tags1,
                COALESCE(m.collab_count_{year}_{month}, 0) AS cc1,
                COALESCE(m.collab_popularity_{year}_{month}, 0) AS cp1,
                COALESCE(m.solo_count_{year}_{month}, 0) AS sc1,
                COALESCE(m.solo_popularity_{year}_{month}, 0) AS sp1,

                lfm,
                mrt,
                prt,
                lt
        """

        x_train = list()
        y_train = list()
        x_test = list()
        y_test = list()

        logging.info("Collecting positive edges...")
        for record in tqdm(session.run(query)):  # type: ignore
            data_list = [
                record["hbd0"],
                record["bd0"],
                record["hed0"],
                record["ed0"],
                record["e0"],
                record["g10"],
                record["g20"],
                record["g30"],
                record["g40"],
                record["g50"],
                record["ps0"],
                record["t10"],
                record["t20"],
                record["t30"],
                record["t40"],
                record["t50"],
                record["t60"],
                *record["tags0"],  # type: ignore
                record["cc0"],
                record["cp0"],
                record["sc0"],
                record["sp0"],

                record["hbd1"],
                record["bd1"],
                record["hed1"],
                record["ed1"],
                record["e1"],
                record["g11"],
                record["g21"],
                record["g31"],
                record["g41"],
                record["g51"],
                record["ps1"],
                record["t11"],
                record["t21"],
                record["t31"],
                record["t41"],
                record["t51"],
                record["t61"],
                *record["tags1"],  # type: ignore
                record["cc1"],
                record["cp1"],
                record["sc1"],
                record["sp1"],

                record["lfm"],
                record["mrt"],
                record["prt"],
                record["lt"]
            ]

            id0 = artist_map[record["id0"]]
            id1 = artist_map[record["id1"]]
            if (id0, id1) in train_edges_set:
                x_train.append(data_list)
                y_train.append(1)
            else:
                x_test.append(data_list)
                y_test.append(1)

        logging.info("Positive sizes:")
        logging.info(f"  x_train: {len(x_train)}, {len(x_train[0])}")
        logging.info(f"  y_train: {len(y_train)}")
        logging.info(f"  x_test: {len(x_test)}, {len(x_test[0])}")
        logging.info(f"  y_test: {len(y_test)}")

    return x_train, y_train, x_test, y_test


def get_negative_info(all_tags: list[str], expected_count: int, train_count: int) -> tuple[list, list, list, list]:
    with driver.session() as session:
        query = f"""
            MATCH (n:Artist)
            WHERE 1 = 1
            AND n.popularity_scaled >= {perc_value}
            WITH
                collect(n.main_id) as ids
            WITH
                ids,
                size(ids) AS n_artists
            WITH
                [_ IN range(0, {int(np.sqrt(expected_count)) + 1}) | toInteger(rand() * n_artists)] AS is,
                [_ IN range(0, {int(np.sqrt(expected_count)) + 1}) | toInteger(rand() * n_artists)] AS js,
                ids
            UNWIND is AS i
            UNWIND js AS j
            WITH
                ids[i] AS id0,
                ids[j] AS id1
            MATCH (n:Artist {{main_id: id0}})
            MATCH (m:Artist {{main_id: id1}})
            WHERE NOT EXISTS((n)-[:COLLAB_WITH]->(m))
            OPTIONAL MATCH (n)-[r0:LAST_FM_MATCH]->(m)
            OPTIONAL MATCH (n)-[r1:MUSICALLY_RELATED_TO]->(m)
            OPTIONAL MATCH (n)-[r2:PERSONALLY_RELATED_TO]->(m)
            OPTIONAL MATCH (n)-[r3:LINKED_TO]->(m)
            WITH
                n,
                m,
                {all_tags} AS all_tags,
                CASE
                  WHEN r0 IS NOT NULL THEN r0.pg_weight
                  ELSE 0
                END AS lfm,
                CASE
                  WHEN r1 IS NOT NULL THEN r1.pg_weight
                  ELSE 0
                END AS mrt,
                CASE
                  WHEN r2 IS NOT NULL THEN r2.pg_weight
                  ELSE 0
                END AS prt,
                CASE
                  WHEN r3 IS NOT NULL THEN r3.pg_weight
                  ELSE 0
                END AS lt
            RETURN
                n.main_id AS id0,
                CASE WHEN n.begin_date IS NULL THEN 0 ELSE 1 END AS hbd0,
                COALESCE(n.begin_date, 0) AS bd0,
                CASE WHEN n.end_date IS NULL THEN 0 else 1 END AS hed0,
                COALESCE(n.end_date, 0) AS ed0,
                n.ended AS e0,
                n.gender_1 AS g10,
                n.gender_2 AS g20,
                n.gender_3 AS g30,
                n.gender_4 AS g40,
                n.gender_5 AS g50,
                n.popularity_scaled AS ps0,
                n.type_1 AS t10,
                n.type_2 AS t20,
                n.type_3 AS t30,
                n.type_4 AS t40,
                n.type_5 AS t50,
                n.type_6 AS t60,
                CASE
                  WHEN size(n.tags) = 0 THEN [tag IN all_tags | 0]
                  ELSE  [tag IN all_tags | toFloat(size([x IN n.tags WHERE x = tag])) / size(n.tags)]
                END AS tags0,
                COALESCE(n.collab_count_{year}_{month}, 0) AS cc0,
                COALESCE(n.collab_popularity_{year}_{month}, 0) AS cp0,
                COALESCE(n.solo_count_{year}_{month}, 0) AS sc0,
                COALESCE(n.solo_popularity_{year}_{month}, 0) AS sp0,

                m.main_id AS id1,
                CASE WHEN m.begin_date IS NULL THEN 0 ELSE 1 END AS hbd1,
                COALESCE(m.begin_date, 0) AS bd1,
                CASE WHEN m.end_date IS NULL THEN 0 else 1 END AS hed1,
                COALESCE(m.end_date, 0) AS ed1,
                m.ended AS e1,
                m.gender_1 AS g11,
                m.gender_2 AS g21,
                m.gender_3 AS g31,
                m.gender_4 AS g41,
                m.gender_5 AS g51,
                m.popularity_scaled AS ps1,
                m.type_1 AS t11,
                m.type_2 AS t21,
                m.type_3 AS t31,
                m.type_4 AS t41,
                m.type_5 AS t51,
                m.type_6 AS t61,
                CASE
                  WHEN size(m.tags) = 0 THEN [tag IN all_tags | 0]
                  ELSE  [tag IN all_tags | toFloat(size([x IN m.tags WHERE x = tag])) / size(m.tags)]
                END AS tags1,
                COALESCE(m.collab_count_{year}_{month}, 0) AS cc1,
                COALESCE(m.collab_popularity_{year}_{month}, 0) AS cp1,
                COALESCE(m.solo_count_{year}_{month}, 0) AS sc1,
                COALESCE(m.solo_popularity_{year}_{month}, 0) AS sp1,

                lfm,
                mrt,
                prt,
                lt

            LIMIT {expected_count};
        """

        x = list()
        y = [0] * expected_count
        for record in tqdm(session.run(query)):  # type: ignore
            x.append([
                record["hbd0"],
                record["bd0"],
                record["hed0"],
                record["ed0"],
                record["e0"],
                record["g10"],
                record["g20"],
                record["g30"],
                record["g40"],
                record["g50"],
                record["ps0"],
                record["t10"],
                record["t20"],
                record["t30"],
                record["t40"],
                record["t50"],
                record["t60"],
                *record["tags0"],  # type: ignore
                record["cc0"],
                record["cp0"],
                record["sc0"],
                record["sp0"],

                record["hbd1"],
                record["bd1"],
                record["hed1"],
                record["ed1"],
                record["e1"],
                record["g11"],
                record["g21"],
                record["g31"],
                record["g41"],
                record["g51"],
                record["ps1"],
                record["t11"],
                record["t21"],
                record["t31"],
                record["t41"],
                record["t51"],
                record["t61"],
                *record["tags1"],  # type: ignore
                record["cc1"],
                record["cp1"],
                record["sc1"],
                record["sp1"],

                record["lfm"],
                record["mrt"],
                record["prt"],
                record["lt"]
            ])

    return x[:train_count], y[:train_count], x[train_count:], y[train_count:]


def main():
    with driver.session() as session:
        query = """
            MATCH (n:Tag)
            WITH COLLECT(DISTINCT n.name) AS all_tags
            RETURN all_tags;
        """
        all_tags = session.run(query).data()[0]["all_tags"]

    x_train_positive, y_train_positive, x_test_positive, y_test_positive = get_positive_info(all_tags)

    x_train_negative, y_train_negative, x_test_negative, y_test_negative = get_negative_info(all_tags, len(x_train_positive) + len(x_test_positive), len(x_train_positive))

    logging.info("Negative sizes:")
    logging.info(f"  x_train: {len(x_train_negative)}, {len(x_train_negative[0])}")
    logging.info(f"  y_train: {len(y_train_negative)}")
    logging.info(f"  x_test: {len(x_test_negative)}, {len(x_test_negative[0])}")
    logging.info(f"  y_test: {len(y_test_negative)}")

    logging.info("Saving...")

    x_train = torch.tensor(x_train_positive + x_train_negative, dtype=torch.float32)
    torch.save(x_train, f"pyg_experiments/ds/mlp{year}{month}{perc}x_train.pt")
    x_test = torch.tensor(x_test_positive + x_test_negative, dtype=torch.float32)
    torch.save(x_test, f"pyg_experiments/ds/mlp{year}{month}{perc}x_test.pt")
    y_train = torch.tensor(y_train_positive + y_train_negative, dtype=torch.float32)
    torch.save(y_train, f"pyg_experiments/ds/mlp{year}{month}{perc}y_train.pt")
    y_test = torch.tensor(y_test_positive + y_test_negative, dtype=torch.float32)
    torch.save(y_test, f"pyg_experiments/ds/mlp{year}{month}{perc}y_test.pt")

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

    for year in [2019, 2021, 2023]:
        for perc in [0, 0.5, 0.75, 0.9]:

            # year = 2019
            logging.info("year: %d", year)
            month = 11
            logging.info("month: %d", month)
            # perc = 0.9
            logging.info("perc: %f", perc)

            train_collab_with = torch.load(f"pyg_experiments/ds/collab_with_{year}_{month}_{perc}.pt")
            train_edges_set = set(map(tuple, train_collab_with.t().tolist()))

            with open("pyg_experiments/ds/artist_map.pkl", "rb") as in_file:
                artist_map = pickle.load(in_file)

            # db connection
            driver = GraphDatabase.driver(f"bolt://{DB_HOST}:{DB_PORT}", auth=basic_auth(DB_USER, DB_PASS))

            with driver.session() as session:
                query = f"MATCH (n:Artist) RETURN apoc.agg.percentiles(n.popularity_scaled, [{perc}]) AS p"
                result = session.run(query)  # type: ignore
                perc_value = result.data()[0]["p"][0]

            main()

            driver.close()

    logging.info("DONE!")
