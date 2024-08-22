#!/usr/bin/env python3


import logging
import requests
from typing import Any, LiteralString
from neo4j import Driver, GraphDatabase, basic_auth
from dotenv import load_dotenv
import os
load_dotenv()


def get_artist_id_from_name(driver: Driver, name: str) -> str | None:
    # Error avoidance
    name = name.replace("!", "")
    name = name.replace("/", "")
    name = name.replace("\"", "")

    query = f"""
        WITH \"{name}\" AS input_name
        CALL db.index.fulltext.queryNodes('artist_names_index', input_name) YIELD node, score
        RETURN node.main_id
        ORDER BY score DESC
        LIMIT 1;
    """
    result = execute_query_return(driver, query)

    if len(result) == 0:
        return None

    return result[0]["node.main_id"]


def update_artist(driver: Driver, main_id: str, listeners: int, playcount: int, similar_artists: dict[str, float], tags: list[str]):
    # Update links
    for name, match in similar_artists.items():
        other_id = get_artist_id_from_name(driver, name)

        if other_id is None:
            continue

        query = f"""
            MATCH (a0:Artist {{main_id:  \"{main_id}\"}}), (a1:Artist {{main_id:  \"{other_id}\"}})
            MERGE (a0)-[l0:LAST_FM_MATCH]->(a1)
                ON CREATE SET l0.weight = {match}
                ON MATCH SET l0.weight = CASE WHEN l0.weight < {match} THEN {match} ELSE l0.weight END
            MERGE (a1)-[l1:LAST_FM_MATCH]->(a0)
                ON CREATE SET l1.weight = {match}
                ON MATCH SET l1.weight = CASE WHEN l1.weight < {match} THEN {match} ELSE l1.weight END
            ;
        """
        execute_query(driver, query)

    # Update individual stats
    query = f"""
        MATCH (a:Artist {{main_id:  \"{main_id}\"}})
        SET
            a.listeners = {listeners},
            a.playcount = {playcount},
            a.tags = {tags},
            a.last_fm_call = true,
            a.in_last_fm = true
        ;
    """
    execute_query(driver, query)


def get_artist_info(artist_name: str, last_fm_api_key: str) -> tuple[bool, dict[str, Any] | None]:
    info = requests.get(f"http://ws.audioscrobbler.com/2.0/?method=artist.getInfo&artist={artist_name}&api_key={last_fm_api_key}&format=json").json()
    if "error" in info:
        error_code = info["error"]
        if error_code == 6:
            logging.info(f"An error with code 6 was found. Seems like {artist_name} was not found in the LastFM's API.")
            return True, None
        error_message = info["message"]
        logging.info(f"An error with code {error_code} happened when trying to get info for {artist_name}.")
        logging.info(f"Error message: {error_message}.")
        return False, None
    artist_info = info["artist"]
    similar = requests.get(f"http://ws.audioscrobbler.com/2.0/?method=artist.getSimilar&artist={artist_name}&api_key={last_fm_api_key}&format=json").json()["similarartists"]["artist"]
    artist_info["similar"] = similar
    return True, artist_info


def get_artists_from_db(driver: Driver, artist_count: int) -> list[dict[str, Any]]:
    query = f"MATCH (n: Artist {{last_fm_call: false}}) RETURN n LIMIT {artist_count};"
    return [r["n"] for r in execute_query_return(driver, query)]


def execute_query_return(driver: Driver, query: LiteralString | str) -> list[dict[str, Any]]:
    with driver.session() as session:
        logging.info(f"Querying '{query}'...")
        result = session.run(query)   # type: ignore
        return result.data()


def execute_query(driver: Driver, query: LiteralString | str) -> None:
    with driver.session() as session:
        logging.info(f"Querying '{query}'...")
        _ = session.run(query)   # type: ignore


def main(driver: Driver, last_fm_api_key: str):
    # Ctrl + c for exit
    # Or artist_count as 0
    while True:
        try:
            # Get a few artists from the Neo4j's DB
            artist_count = int(input("How many artists do you want to query? "))

            if artist_count < 0:
                raise ValueError("Please select a valid number of artists. Got a negative number.")

        # Error handling (or not handling)
        except ValueError as e:
            logging.info(e)
            continue
        except Exception as e:
            raise e

        if artist_count == 0:
            logging.info("Exiting...")
            return

        # For each node get the list of names
        for artist in get_artists_from_db(driver, artist_count):
            main_id = artist["main_id"]
            names = artist["known_names"]
            logging.info(f"FOUND ARTIST WITH main_id: '{main_id}'")
            logging.info("Looping through its names...")

            in_db = False
            listeners = 0
            playcount = 0
            similar_artists: dict[str, float] = dict()
            tags: set[str] = set()

            for name in names:
                # For each name get the data from LastFM
                logging.info(f"  Name: {name}")
                success, artist_info = get_artist_info(name, last_fm_api_key)

                # What happens if fail?
                #     I'll just print what happened and check it out later,
                #     I should not query for too many artists anyways (battery life and/or API's rate limit)
                #     Then I'll see how to handle that error
                if not success:
                    logging.critical("UNEXPECTED ERROR!")
                    exit(1)

                # Not found in the API
                if success and artist_info is None:
                    in_db = in_db or False
                    continue

                # Found!
                in_db = True

                # We should be good to go to extract the artist's info
                if (stats := artist_info.get("stats", False)):
                    listeners += int(stats["listeners"])
                    playcount += int(stats["playcount"])
                    logging.info(f"    Listeners: {listeners}")
                    logging.info(f"    Playcount: {playcount}")

                if (artist_tags := artist_info.get("tags", False)):
                    if (artist_tags := artist_tags.get("tag", False)):
                        tag_list = list(tag["name"] for tag in artist_tags)
                        tags.update(tag_list)
                        logging.info(f"    Found tags: {tag_list}")

                for similar_artist in artist_info.get("similar", []):
                    similar_artist: dict[str, Any]
                    similar_artist_name = similar_artist["name"]
                    match = max(
                        float(similar_artist["match"]),
                        similar_artists.get(similar_artist_name, 0)
                    )
                    similar_artists[similar_artist_name] = match
                    logging.info(f"    Found similar artist: '{similar_artist_name}'. Match: '{match}'")

            if in_db:
                tags = list(tags)
                logging.info(f"Updating artist {main_id} with {listeners} listeners, {playcount} playcount, {len(similar_artists)} similar artists and {len(tags)} tags...")
                update_artist(driver, main_id, listeners, playcount, similar_artists, tags)

            # NOT IN API -> no data could be extracted
            else:
                logging.info(f"  Seems like we couldn't extract info for artist {main_id}...")
                query = f"MATCH (a:Artist {{main_id:  \"{main_id}\"}}) SET a.last_fm_call = true, a.in_last_fm = false"
                execute_query(driver, query)


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
