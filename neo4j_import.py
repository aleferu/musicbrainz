#!/usr/bin/env python3


"""
This script will import all of our current information into a neo4j database.
"""


from typing import LiteralString
from neo4j import Driver, GraphDatabase, basic_auth
from dotenv import load_dotenv
import os
load_dotenv()


def execute_query(driver: Driver, query: LiteralString | str, print_records: bool = False):
    with driver.session() as session:
        print(f"Querying '{query}'...")
        result = session.run(query)  # type: ignore
        if print_records:
            print("Printing records:")
            for record in result:
                print(record)


def remove_db(driver: Driver):
    query = """
        CALL apoc.periodic.iterate(
          "MATCH (n) RETURN n",
          "DETACH DELETE n",
          {batchSize: 100000}
        );
    """
    execute_query(driver, query, True)


def import_mb_tags(driver: Driver):
    query = "CREATE INDEX IF NOT EXISTS FOR (t:MBTag) ON (t.id);"
    execute_query(driver, query)

    query = """
        CALL apoc.periodic.iterate(
            "LOAD CSV WITH HEADERS FROM 'file:///tags.csv' AS row RETURN row",
            "MERGE (t:MBTag {id: row.id, name: row.name})",
            {batchSize: 10000, parallel: true, concurrency: 8}
        );
    """
    execute_query(driver, query, True)


def import_artists(driver: Driver):
    # dbms.security.procedures.unrestricted=apoc.*
    # dbms.security.procedures.allowlist=apoc.*
    # If stuck: google how to use apoc for loading JSON files in Neo4j
    # It's really simple, moving a .jar file and a few settings
    # You also need to "cp artists.jsonl /var/lib/neo4j/import/artists.jsonl"
    query = "CREATE FULLTEXT INDEX artist_names_index IF NOT EXISTS FOR (a:Artist) ON EACH [a.known_names];"
    execute_query(driver, query)

    query = "CREATE INDEX IF NOT EXISTS FOR (a:Artist) ON (a.main_id);"
    execute_query(driver, query)

    query = """
        CALL apoc.periodic.iterate(
            "CALL apoc.load.json('file:///artists.jsonl') YIELD value",
            "MERGE (a:Artist {
                main_id: value.main_id,
                known_ids: value.known_ids,
                known_names: value.known_names,
                listeners: 0,
                playcount: 0,
                last_fm_call: false,
                in_last_fm: false
            })",
            {batchSize: 10000, parallel: true, concurrency: 8}
        );
    """
    execute_query(driver, query, True)


def create_artist_mbtag_links(driver: Driver):
    query = """
        CALL apoc.periodic.iterate(
            "LOAD CSV WITH HEADERS FROM 'file:///artist_tags.csv' AS row RETURN row",
            "
                WITH
                    row.artist AS artist_id,
                    SPLIT(row.tags, ', ') AS tag_ids
                UNWIND tag_ids AS tag_id
                MATCH (a:Artist {main_id: artist_id}), (t:MBTag {id: tag_id})
                MERGE (a)-[:HAS_TAG]->(t)
                MERGE (t)-[:TAGS]->(a)
            ",
            {batchSize: 10000, parallel: false}
        );
    """
    execute_query(driver, query, True)


def import_releases(driver: Driver):
    query = "CREATE INDEX IF NOT EXISTS FOR (r:Release) ON (r.id);"
    execute_query(driver, query)

    query = """
        CALL apoc.periodic.iterate(
            "LOAD CSV WITH HEADERS FROM 'file:///releases_no_va_merged_id.csv' AS row RETURN row",
            "
                MERGE (r:Release {id: row.id})
                ON CREATE SET
                    r.name = row.name,
                    r.date = DATE(row.date),
                    r.artist_count = row.artist_count,
                    r.listeners = 0,
                    r.playcount = 0,
                    r.last_fm_call = false,
                    r.in_last_fm = false
                WITH
                    [row.a0_id, row.a1_id, row.a2_id, row.a3_id, row.a4_id] AS artist_ids,
                    r
                UNWIND artist_ids AS artist_id
                MATCH (a:Artist {main_id: artist_id})
                MERGE (a)-[:WORKED_IN]->(r)
                MERGE (r)-[:WORKED_BY]->(a)
            ",
            {batchSize: 50000}
        );
    """
    execute_query(driver, query, True)

    query = """
        CALL apoc.periodic.iterate(
            "LOAD CSV WITH HEADERS FROM 'file:///releases_no_va_merged_id.csv' AS row RETURN row",
            "
                MATCH (r:Release {id: row.id})
                WITH r, SPLIT(r.tags, ', ') AS release_tags
                UNWIND release_tags AS release_tag
                MATCH (t:MBTag {id: release_tag})
                MERGE (r)-[:HAS_TAG]->(t)
                MERGE (t)-[:TAGS]->(r)
            ",
            {batchSize: 50000}
        );
    """
    execute_query(driver, query, True)


def add_coll_links(driver: Driver):
    query = """
        MATCH (a0:Artist)-[:WORKED_IN]->(r:Release)<-[:WORKED_IN]-(a1:Artist)
        WHERE a0.main_id < a1.main_id
        WITH a0, a1
        MERGE (a0)-[c0:COLLAB_WITH]->(a1)
            ON CREATE SET c0.count = 1
            ON MATCH SET c0.count = c0.count + 1
        MERGE (a1)-[c1:COLLAB_WITH]->(a0)
            ON CREATE SET c1.count = 1
            ON MATCH SET c1.count = c1.count + 1
        ;
    """
    execute_query(driver, query)


def add_relationships_links(driver: Driver):
    relationship_mappings = [
        {"types": [102, 103, 104, 105, 106, 107, 108, 305, 728, 855, 965], "label": "MUSICALLY_RELATED_TO"},
        {"types": [109, 110, 111, 112, 113, 292, 973, 1079], "label": "PERSONALLY_RELATED_TO"},
        {"types": [722, 847, 895], "label": "LINKED_TO"},
    ]

    for mapping in relationship_mappings:
        types_list = str(mapping["types"])
        relationship_label = mapping["label"]

        # Must be a faster way but it works so good enough for now
        # It's also fast so might never change it
        query = f"""
            CALL apoc.periodic.iterate(
                "
                    LOAD CSV WITH HEADERS FROM 'file:///relationships_clean.csv' AS row
                    RETURN row
                ",
                "
                    MATCH (a0:Artist {{main_id: row.id0}}), (a1:Artist {{main_id: row.id1}})
                    WITH a0, a1, toInteger(row.relationship_type) as rel_type
                    WHERE rel_type IN {types_list}
                    WITH a0, a1
                    MERGE (a0)-[rel:{relationship_label}]->(a1)
                        ON CREATE SET rel.count = 1
                        ON MATCH SET rel.count = rel.count + 1
                    MERGE (a1)-[rel_back:{relationship_label}]->(a0)
                        ON CREATE SET rel_back.count = 1
                        ON MATCH SET rel_back.count = rel_back.count + 1
                ",
                {{batchSize: 10000}}
            );
        """
        execute_query(driver, query, True)


def main(driver: Driver) -> None:
    # Remove last iteration of the db
    remove_db(driver)

    # First let's import the tags obtained from MB
    import_mb_tags(driver)

    # Then, let's import our artist database
    import_artists(driver)

    # With their tags
    create_artist_mbtag_links(driver)

    # Now the releases
    import_releases(driver)

    # Create the collaboration links
    add_coll_links(driver)

    # Create the relationships links
    add_relationships_links(driver)


if __name__ == '__main__':
    # .env read
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

    # cleanup
    driver.close()
