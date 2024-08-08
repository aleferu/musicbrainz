#!/usr/bin/env python3


"""
This script will import all of our current information into a neo4j database.
"""


from typing import LiteralString
from neo4j import Driver, GraphDatabase, basic_auth


def execute_and_print_query(driver: Driver, query: LiteralString):
    with driver.session() as session:
        result = session.run(query)
        for record in result:
            print(record)


def import_artists(driver: Driver):
    # dbms.security.procedures.unrestricted=apoc.*
    # dbms.security.procedures.allowlist=apoc.*
    # If stuck: google how to use apoc for loading JSON files in Neo4j
    # It's really simple, moving a .jar file and a few settings
    # You also need to "cp artists.jsonl /var/lib/neo4j/import/artists.jsonl"
    query = """
        CALL apoc.periodic.iterate(
            "CALL apoc.load.json('file:///artists.jsonl') YIELD value",
            "MERGE (n:Artist {main_id: value.main_id, known_ids: value.known_ids, known_names: value.known_names})",
            {batchSize: 10000, parallel: true, concurrency: 8}
        );
    """
    execute_and_print_query(driver, query)

    query = "CREATE INDEX IF NOT EXISTS FOR (a:Artist) ON (a.main_id);"
    execute_and_print_query(driver, query)


def import_releases(driver: Driver):
    query = """
        CALL apoc.periodic.iterate(
            "LOAD CSV WITH HEADERS FROM 'file:///releases_no_va_merged_id.csv' AS row RETURN row",
            "
                WITH
                    [row.a0_id, row.a1_id, row.a2_id, row.a3_id, row.a4_id] AS ids,
                    row
                UNWIND ids AS id0
                UNWIND ids AS id1
                MATCH (a0:Artist {main_id: id0}), (a1:Artist {main_id: id1})
                WHERE id0 < id1
                WITH
                    a0,
                    a1,
                    row,
                    date(row.date) AS coll_date
                MERGE (a0)-[r:COLLABORATED_WITH]->(a1)
                    ON CREATE SET
                      r.names = [row.name],
                      r.dates = [coll_date],
                      r.ids = [row.id]
                    ON MATCH SET
                      r.names = r.names + [row.name],
                      r.dates = r.dates + [coll_date],
                      r.ids = r.ids + [row.id]
            ",
            {batchSize: 50000, parallel: false, concurrency: 8}
        );
    """
    execute_and_print_query(driver, query)


def main(driver: Driver) -> None:
    # First let's import our artist database
    import_artists(driver)

    # Now the releases
    import_releases(driver)


if __name__ == '__main__':
    # db connection
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=basic_auth("neo4j", ""))

    main(driver)

    # cleanup
    driver.close()
