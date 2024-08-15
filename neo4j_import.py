#!/usr/bin/env python3


"""
This script will import all of our current information into a neo4j database.
"""


from typing import LiteralString
from neo4j import Driver, GraphDatabase, basic_auth


def execute_query(driver: Driver, query: LiteralString, print_records: bool = False):
    with driver.session() as session:
        print(f"Querying '{query}'...")
        result = session.run(query)
        if print_records:
            print("Printing records:")
            for record in result:
                print(record)


def import_artists(driver: Driver):
    # dbms.security.procedures.unrestricted=apoc.*
    # dbms.security.procedures.allowlist=apoc.*
    # If stuck: google how to use apoc for loading JSON files in Neo4j
    # It's really simple, moving a .jar file and a few settings
    # You also need to "cp artists.jsonl /var/lib/neo4j/import/artists.jsonl"
    query = "CREATE INDEX IF NOT EXISTS FOR (a:Artist) ON (a.main_id);"
    execute_query(driver, query)

    query = """
        CALL apoc.periodic.iterate(
            "CALL apoc.load.json('file:///artists.jsonl') YIELD value",
            "MERGE (n:Artist {
                main_id: value.main_id,
                known_ids: value.known_ids,
                known_names: value.known_names,
                tags: [],
                listeners: 0,
                playcount: 0
            })",
            {batchSize: 10000, parallel: true, concurrency: 8}
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
                WITH
                    [row.a0_id, row.a1_id, row.a2_id, row.a3_id, row.a4_id] AS ids,
                    row
                UNWIND ids AS artist_id
                WITH
                    artist_id,
                    row
                MATCH (a:Artist {main_id: artist_id})
                WITH
                    a,
                    row,
                    date(row.date) AS coll_date
                MERGE (r:Release {id: row.id})
                ON CREATE SET r.name = row.name, r.date = coll_date, r.artist_count = row.artist_count
                MERGE (a)-[:WORKED_IN]->(r)
                MERGE (r)-[:WORKED_BY]->(a)
            ",
            {batchSize: 50000, parallel: true, concurrency: 8}
        );
    """
    execute_query(driver, query, True)


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
