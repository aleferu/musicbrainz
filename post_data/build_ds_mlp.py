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
            WITH
                n,
                m,
                {all_tags} AS all_tags
            RETURN
                n.main_id AS id0,
                COALESCE(n.begin_date, -1) AS bd0,
                COALESCE(n.end_date, -1) AS ed0,
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

                m.main_id AS id1,
                COALESCE(m.begin_date, -1) AS bd1,
                COALESCE(m.end_date, -1) AS ed1,
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
                END AS tags1
        """

        x_train = list()
        y_train = list()
        x_test = list()
        y_test = list()

        logging.info("Collecting positive edges...")
        for record in tqdm(session.run(query)):  # type: ignore
            data_list = [
                record["bd0"],
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

                record["bd1"],
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
            WITH
                n,
                m,
                {all_tags} AS all_tags
            RETURN
                n.main_id AS id0,
                COALESCE(n.begin_date, -1) AS bd0,
                COALESCE(n.end_date, -1) AS ed0,
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

                m.main_id AS id1,
                COALESCE(m.begin_date, -1) AS bd1,
                COALESCE(m.end_date, -1) AS ed1,
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
                END AS tags1
            LIMIT {expected_count};
        """

        x = list()
        y = [0] * expected_count
        for record in tqdm(session.run(query)):  # type: ignore
            x.append([
                record["bd0"],
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

                record["bd1"],
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

    assert len(x_train_negative[0]) == len(x_train_positive[0])
    assert len(x_train_negative) == len(x_train_positive)
    assert len(x_test_negative[0]) == len(x_test_positive[0])
    assert len(x_test_negative) == len(x_test_positive)
    assert len(y_train_positive) == len(y_train_negative)
    assert len(y_test_positive) == len(y_test_negative)

    logging.info("Saving...")

    x_train = torch.tensor(x_train_positive + x_train_negative, dtype=torch.float32)
    torch.save(x_train, "post_data/ds/mlp20200300x_train.pt")
    x_test = torch.tensor(x_test_positive + x_test_negative, dtype=torch.float32)
    torch.save(x_test, "post_data/ds/mlp20200300x_test.pt")
    y_train = torch.tensor(y_train_positive + y_train_negative, dtype=torch.float32)
    torch.save(y_train, "post_data/ds/mlp20200300y_train.pt")
    y_test = torch.tensor(y_test_positive + y_test_negative, dtype=torch.float32)
    torch.save(y_test, "post_data/ds/mlp20200300y_test.pt")

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

    train_collab_with = torch.load("pyg_experiments/ds/collab_with_2020_3.pt")
    train_edges_set = set(map(tuple, train_collab_with.t().tolist()))

    with open("pyg_experiments/ds/artist_map.pkl", "rb") as in_file:
        artist_map = pickle.load(in_file)

    # db connection
    driver = GraphDatabase.driver(f"bolt://{DB_HOST}:{DB_PORT}", auth=basic_auth(DB_USER, DB_PASS))

    main()

    driver.close()

    logging.info("DONE!")
