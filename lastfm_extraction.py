#!/usr/bin/env python3


import requests
from typing import Any, LiteralString
from neo4j import Driver, GraphDatabase, basic_auth
from dotenv import load_dotenv
import os
load_dotenv()


def get_artist_info(artist_name: str, last_fm_api_key: str) -> dict[str, Any] | None:
    info = requests.get(f"http://ws.audioscrobbler.com/2.0/?method=artist.getInfo&artist={artist_name}&api_key={last_fm_api_key}&format=json").json()
    if "error" in info:
        error_code = info["error"]
        error_message = info["message"]
        print(f"An error with code {error_code} happened when trying to get info for {artist_name}.")
        print(f"Error message: {error_message}.")
        return None
    artist_info = info["artist"]
    similar = requests.get(f"http://ws.audioscrobbler.com/2.0/?method=artist.getSimilar&artist={artist_name}&api_key={last_fm_api_key}&format=json").json()["similarartists"]["artist"]
    artist_info["similar"] = similar
    return artist_info


def execute_query(driver: Driver, query: LiteralString | str, print_records: bool = False):
    with driver.session() as session:
        print(f"Querying '{query}'...")
        result = session.run(query)
        if print_records:
            print("Printing records:")
            for record in result:
                print(record)


def main(driver: Driver):
    # Get a few artists from the Neo4j's DB
    # For each node get the list of names
    # For each name get the data from LastFM
    #   What happens if fail?
    #   Do I just import everything or the "best" found account?
    # Create the links
    # Somehow make it so that I can stop/resume anytime I want
    pass


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
