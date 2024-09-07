#!/usr/bin/env python3


import logging
import requests_async as requests
from typing import Any, LiteralString
from neo4j import AsyncDriver, AsyncGraphDatabase, basic_auth
import neo4j.exceptions as neo4j_exceptions
from dotenv import load_dotenv
import os
import asyncio


async def get_artist_id_from_name(driver: AsyncDriver, name: str) -> str | None:
    # Error avoidance
    name = name.replace("!", "")
    name = name.replace("/", "")
    name = name.replace("\"", "")
    name = name.replace("[", "")
    name = name.replace("]", "")
    name = name.replace("(", "")
    name = name.replace(")", "")
    name = name.replace(":", "")
    name = name.replace("₩", "")
    name = name.replace("-", "")
    name = name.replace("~", "")
    name = name.replace("{", "")
    name = name.replace("}", "")
    name = name.replace("^", "")

    if len(name.strip()) == 0:
        return None

    query = f"""
        WITH \"{name}\" AS input_name
        CALL db.index.fulltext.queryNodes('artist_names_index', input_name) YIELD node, score
        WHERE score > 1
        RETURN node.main_id
        ORDER BY score DESC
        LIMIT 1;
    """
    try:
        result = await execute_query_return(driver, query)
    except neo4j_exceptions.ClientError as e:
        logging.critical(f"ERROR: {e}")
        logging.critical(f"Name: {name}")
        exit()

    if len(result) == 0:
        return None

    return result[0]["node.main_id"]


async def get_tag_id_from_name(driver: AsyncDriver, name: str) -> str | None:
    query = f"""
        WITH \"{name}\" AS input_name
        CALL db.index.fulltext.queryNodes('last_fm_tag_names_index', input_name) YIELD node, score
        WHERE score > 1
        RETURN node.name, node.id
        ORDER BY score DESC
        LIMIT 1;
    """
    result = await execute_query_return(driver, query)

    if len(result) == 0:
        return None

    return result[0]["node.id"]


async def update_artist(driver: AsyncDriver, main_id: str, listeners: int, playcount: int, similar_artists: dict[str, float], tags: list[str]):
    # Update tags
    for tag_name in tags:
        tag_id = await get_tag_id_from_name(driver, tag_name)

        if tag_id is None:
            query = f"""
                MATCH (a:Artist {{main_id: \"{main_id}\"}})
                MERGE (t:LFMTag {{id: randomUUID(), name: \"{tag_name}\"}})
                MERGE (t)-[:TAGS]->(a)
                MERGE (a)-[:HAS_TAG]->(t)
            """
        else:
            query = f"""
                MATCH (a:Artist {{main_id: \"{main_id}\"}}), (t:LFMTag {{id: \"{tag_id}\", name: \"{tag_name}\"}})
                MERGE (t)-[:TAGS]->(a)
                MERGE (a)-[:HAS_TAG]->(t)
            """
        _ = await execute_query(driver, query)

    # Update links
    for name, match in similar_artists.items():
        other_id = await get_artist_id_from_name(driver, name)

        if other_id is None:
            continue

        query = f"""
            MATCH (a0:Artist {{main_id: \"{main_id}\"}}), (a1:Artist {{main_id: \"{other_id}\"}})
            MERGE (a0)-[l0:LAST_FM_MATCH]->(a1)
                ON CREATE SET l0.weight = {match}
                ON MATCH SET l0.weight = CASE WHEN l0.weight < {match} THEN {match} ELSE l0.weight END
            MERGE (a1)-[l1:LAST_FM_MATCH]->(a0)
                ON CREATE SET l1.weight = {match}
                ON MATCH SET l1.weight = CASE WHEN l1.weight < {match} THEN {match} ELSE l1.weight END
            ;
        """
        _ = await execute_query(driver, query)

    # Update individual stats
    query = f"""
        MATCH (a:Artist {{main_id:  \"{main_id}\"}})
        SET
            a.listeners = {listeners},
            a.playcount = {playcount},
            a.last_fm_call = true,
            a.in_last_fm = true
        ;
    """
    _ = await execute_query(driver, query)


async def get_artist_info(artist_name: str, last_fm_api_key: str) -> tuple[bool, dict[str, Any] | None]:
    request_url = f"http://ws.audioscrobbler.com/2.0/?method=artist.getInfo&artist={artist_name}&autocorrect=1&api_key={last_fm_api_key}&format=json"
    info = await requests.get(request_url)

    if info.status_code != 200:
        logging.info(f"FOUND ERROR CODE: {info.status_code}")
        logging.info(request_url)
        logging.info("Request text in next line:")
        logging.info(info.text)
        exit()

    info = info.json()
    if "error" in info:
        error_code = info["error"]
        if error_code == 6:
            logging.error(f"An error with code 6 was found. Seems like {artist_name} was not found in the LastFM's API.")
            return True, None
        error_message = info["message"]
        logging.error(f"An error with code {error_code} happened when trying to get info for {artist_name}.")
        logging.error(f"Error message: {error_message}.")
        return False, None
    artist_info = info["artist"]
    similar = await requests.get(f"http://ws.audioscrobbler.com/2.0/?method=artist.getSimilar&artist={artist_name}&autocorrect=1&api_key={last_fm_api_key}&format=json")
    similar = similar.json()["similarartists"]["artist"]
    artist_info["similar"] = similar
    return True, artist_info


async def get_artists_from_db(driver: AsyncDriver, artist_count: int) -> list[dict[str, Any]]:
    query = f"MATCH (n: Artist {{last_fm_call: false}}) RETURN n LIMIT {artist_count};"
    query_result = await execute_query_return(driver, query)
    return [r["n"] for r in query_result]


async def execute_query_return(driver: AsyncDriver, query: LiteralString | str) -> list[dict[str, Any]]:
    async with driver.session() as session:
        logging.info(f"Querying '{query}'...")
        result = await session.run(query)  # type: ignore
        return await result.data()


async def execute_query(driver: AsyncDriver, query: LiteralString | str) -> None:
    async with driver.session() as session:
        logging.info(f"Querying '{query}'...")
        _ = await session.run(query)   # type: ignore


async def process_artist(driver: AsyncDriver, artist: dict[str, Any], last_fm_api_key: str):
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
        success, artist_info = await get_artist_info(name, last_fm_api_key)

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

        assert artist_info is not None, "LSP was complaining"

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
        tag_list = list(tags)
        logging.info(f"Updating artist {main_id} with {listeners} listeners, {playcount} playcount, {len(similar_artists)} similar artists and {len(tag_list)} tags...")
        _ = await update_artist(driver, main_id, listeners, playcount, similar_artists, tag_list)

    # NOT IN API -> no data could be extracted
    else:
        logging.error(f"  Seems like we couldn't extract info for artist {main_id}...")
        query = f"MATCH (a:Artist {{main_id:  \"{main_id}\"}}) SET a.last_fm_call = true, a.in_last_fm = false"
        _ = await execute_query(driver, query)


async def main(driver: AsyncDriver, last_fm_api_key: str):
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
            logging.info("An error has occured.")
            logging.info(e)
            return

        if artist_count == 0:
            logging.info("Exiting...")
            return

        # Process each artist
        _ = await asyncio.gather(*[process_artist(driver, artist, last_fm_api_key) for artist in await get_artists_from_db(driver, artist_count)])


if __name__ == '__main__':
    load_dotenv()

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
    driver = AsyncGraphDatabase.driver(f"bolt://{DB_HOST}:{DB_PORT}", auth=basic_auth(DB_USER, DB_PASS))

    _ = asyncio.run(main(driver, LAST_FM_API_KEY))

    # cleanup
    _ = asyncio.run(driver.close())
