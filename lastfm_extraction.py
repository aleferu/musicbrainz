#!/usr/bin/env python3


import logging
from neo4j._work import query
import requests
from typing import Any, LiteralString
from neo4j import Driver, GraphDatabase, Record, basic_auth
from dotenv import load_dotenv
import os
load_dotenv()


def get_artist_info(artist_name: str, last_fm_api_key: str) -> dict[str, Any] | None:
    info = requests.get(f"http://ws.audioscrobbler.com/2.0/?method=artist.getInfo&artist={artist_name}&api_key={last_fm_api_key}&format=json").json()
    if "error" in info:
        error_code = info["error"]
        error_message = info["message"]
        logging.info(f"An error with code {error_code} happened when trying to get info for {artist_name}.")
        logging.info(f"Error message: {error_message}.")
        return None
    artist_info = info["artist"]
    similar = requests.get(f"http://ws.audioscrobbler.com/2.0/?method=artist.getSimilar&artist={artist_name}&api_key={last_fm_api_key}&format=json").json()["similarartists"]["artist"]
    artist_info["similar"] = similar
    return artist_info


def get_artists_from_db(driver: Driver, artist_count: int) -> list[dict[str, Any]]:
    query = f"MATCH (n: Artist {{last_fm_call: false}}) RETURN n LIMIT {artist_count}"
    return [r["n"] for r in execute_query_return(driver, query)]


def execute_query_return(driver: Driver, query: LiteralString | str) -> list[Record]:
    with driver.session() as session:
        logging.info(f"Querying '{query}'...")
        result = session.run(query)   # type: ignore
        return [r for r in result]


def execute_query(driver:Driver, query: LiteralString | str) -> None:
    with driver.session() as session:
        logging.info(f"Querying '{query}'...")
        result = session.run(query)   # type: ignore


def main(driver: Driver, last_fm_api_key: str):
    # Somehow make it so that I can stop/resume anytime I want
    #   Done with input and a node variable
    # Ctrl + c for exit
    while True:
        try:
            # Get a few artists from the Neo4j's DB
            artist_count = int(input("How many artists do you want to query? "))

            if artist_count == 0:
                logging.info("Exiting...")
                return

            # For each node get the list of names
            for artist in get_artists_from_db(driver, artist_count):
                main_id = artist["main_id"]
                names = artist["known_names"]
                logging.info(f"FOUND ARTIST WITH main_id: '{main_id}'")
                logging.info("Looping through its names...")

                for name in names:
                    # For each name get the data from LastFM
                    logging.info(f"  Name: {name}")
                    artist_info = get_artist_info(name, last_fm_api_key)

                    # What happens if fail?
                    #     I'll just print what happened and check it out later,
                    #     I should not query for too many artists anyways (battery life and/or API's rate limit)
                    if artist_info is None:
                        continue
                    
                    # Do I just import everything or the "best" found account?
                    #     I'll just sum everything up, what could go wrong?

                    # TODO: set in_last_fm to true in db

                    if (stats := artist_info.get("stats", False)):
                        listeners = int(stats["listeners"])
                        playcount = int(stats["playcount"])
                        logging.info(f"    Listeners: {listeners}")
                        logging.info(f"    Playcount: {playcount}")

                        # TODO: update listeners and playcount

                    for similar_artist in artist_info.get("similar", []):
                        similar_artist: dict[str, Any]
                        similar_artist_name = similar_artist["name"]
                        match = float(similar_artist["match"])
                        logging.info(f"    Found similar artist: '{similar_artist_name}'. Match: '{match}'")

                        # TODO: create link

                    # TODO: set last_fm_call in db

        # Error handling (or not handling)
        except ValueError as e:
            logging.info(e)
        except Exception as e:
            raise e


if __name__ == '__main__':
    # logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


    # .env read
    DB_HOST = os.getenv("NEO4J_HOST")
    DB_PORT = os.getenv("NEO4J_PORT")
    DB_USER = os.getenv("NEO4J_USER")
    DB_PASS = os.getenv("NEO4J_PASS")
    LAST_FM_API_KEY = os.getenv("LAST_FM_API_KEY")

    # .env validation
    assert DB_HOST is not None and \
        DB_PORT is not None and \
        DB_USER is not None and \
        DB_PASS is not None and \
        LAST_FM_API_KEY is not None, \
        "INVALID .env"

    # db connection
    driver = GraphDatabase.driver(f"bolt://{DB_HOST}:{DB_PORT}", auth=basic_auth(DB_USER, DB_PASS))

    main(driver, LAST_FM_API_KEY)

    # cleanup
    driver.close()
