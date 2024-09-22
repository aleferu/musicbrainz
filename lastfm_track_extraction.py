#!/usr/bin/env python3


import urllib.parse
import logging
import requests_async as requests
from typing import Any
from neo4j import AsyncDriver, AsyncGraphDatabase, basic_auth
from dotenv import load_dotenv
import os
import asyncio

from lastfm_artist_extraction import get_tag_mapping, get_tag_ids, execute_query, execute_query_return


async def update_release(driver: AsyncDriver, release_id: str, listeners: int, playcount: int, tags: list[str], tag_mapping: dict[str, set[str]]):
    # Update tags
    tag_ids = get_tag_ids(tags, tag_mapping)
    if len(tag_ids) > 0:
        query = f"""
            MATCH (r:Release {{id: \"{release_id}\"}})
            UNWIND {str(tag_ids)} AS tag_id
            MATCH (t:Tag {{id: tag_id}})
            MERGE (t)-[:TAGS]->(r)
            MERGE (r)-[:HAS_TAG]->(t)
        """
        _ = await execute_query(driver, query)

    # Update individual stats
    query = f"""
        MATCH (r:Release {{id:  \"{release_id}\"}})
        SET
            r.listeners = {listeners},
            r.playcount = {playcount},
            r.last_fm_call = true,
            r.in_last_fm = true
        ;
    """
    _ = await execute_query(driver, query)


async def get_release_info(release_name: str, artist_name: str, last_fm_api_key: str) -> tuple[bool, dict[str, Any] | None]:
    release_name_clean = urllib.parse.quote(release_name)
    artist_name_clean = urllib.parse.quote(artist_name)
    request_url = f"http://ws.audioscrobbler.com/2.0/?method=track.getInfo&artist={artist_name_clean}&track={release_name_clean}&autocorrect=1&api_key={last_fm_api_key}&format=json"
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
            logging.error(f"An error with code 6 was found. Seems like the combination of {release_name} and {artist_name} was not found in the LastFM's API.")
            return True, None
        error_message = info["message"]
        logging.error(f"An error with code {error_code} happened when trying to get info for the combination of {release_name} and {artist_name}.")
        logging.error(f"Error message: {error_message}.")
        return False, None
    release_info = info["track"]

    top_tags = await requests.get(f"http://ws.audioscrobbler.com/2.0/?method=track.getTopTags&track={release_name_clean}&artist={artist_name_clean}&autocorrect=1&api_key={last_fm_api_key}&format=json")
    top_tags = top_tags.json()
    if "toptags" in top_tags:
        top_tags["toptags"]["tag"] = list(filter(lambda t: t["count"] >= 5, top_tags["toptags"]["tag"]))  # 5 seems ok
        release_info["tags"] = top_tags["toptags"]

    return True, release_info


async def get_artists_from_release(driver: AsyncDriver, release_id: str) -> list[str]:
    query = f"MATCH (r:Release {{id: \"{release_id}\"}})-[]->(a:Artist) UNWIND(a.known_names) AS names RETURN COLLECT(names) AS names;"
    result = await execute_query_return(driver, query)
    return result[0]["names"]


async def get_releases_from_db(driver: AsyncDriver, release_count: int) -> list[dict[str, Any]]:
    query = f"MATCH (n: Release {{last_fm_call: false}}) RETURN n LIMIT {release_count};"
    query_result = await execute_query_return(driver, query)
    return [r["n"] for r in query_result]


async def process_release(driver: AsyncDriver, release: dict[str, Any], last_fm_api_key: str, tag_mapping: dict[str, set[str]]):
    release_id = release["id"]
    release_name = release["name"]
    logging.info(f"FOUND ARTIST WITH id '{release_id}' and name '{release_name}'")

    artists = await get_artists_from_release(driver, release_id)
    logging.info("Looping through its artists...")

    in_db = False
    listeners = 0
    playcount = 0
    tags: set[str] = set()

    for artist_name in artists:
        logging.info(f"  Trying with the artist name {artist_name}")
        success, release_info = await get_release_info(release_name, artist_name, last_fm_api_key)

        if not success:
            logging.critical("UNEXPECTED ERROR!")
            exit(1)

        if success and release_info is None:
            in_db = in_db or False
            continue

        assert release_info is not None, "LSP was complaining"

        in_db = True

        listeners += int(release_info.get("listeners", "0"))
        logging.info(f"    Listeners: {listeners}")
        playcount += int(release_info.get("playcount", "0"))
        logging.info(f"    Playcount: {playcount}")

        if (release_tags := release_info.get("toptags", False)):
            if (release_tags := release_tags.get("tag", False)):
                tag_list = list(tag["name"] for tag in release_tags)
                tags.update(tag_list)
                logging.info(f"    Found tags: {tag_list}")

    if in_db:
        tag_list = list(tags)
        logging.info(f"Updating release {release_id} with {listeners} listeners, {playcount} playcount, and {len(tag_list)} tags...")
        _ = await update_release(driver, release_id, listeners, playcount, tag_list, tag_mapping)

    else:
        logging.error(f"  Seems like we couldn't extract info for release {release_id}...")
        query = f"MATCH (r:Release {{id: \"{release_id}\"}}) SET r.last_fm_call = true, r.in_last_fm = false"
        _ = await execute_query(driver, query)


async def main(driver: AsyncDriver, last_fm_api_key: str):
    tag_mapping = get_tag_mapping()

    if tag_mapping is None:
        return

    # Ctrl + c for exit
    # Or release_count as 0
    while True:
        try:
            # Get a few releases from the Neo4j's DB
            release_count = int(input("How many releases do you want to query? "))

            if release_count < 0:
                raise ValueError("Please select a valid number of releases. Got a negative number.")

        # Error handling (or not handling)
        except ValueError as e:
            logging.info(e)
            continue
        except Exception as e:
            logging.info("An error has occured.")
            logging.info(e)
            return

        if release_count == 0:
            logging.info("Exiting...")
            return

        # Process each artist
        _ = await asyncio.gather(*[process_release(driver, release, last_fm_api_key, tag_mapping.copy()) for release in await get_releases_from_db(driver, release_count)])


async def run_and_clean(driver: AsyncDriver, last_fm_api_key: str):
    _ = await main(driver, last_fm_api_key)

    _ = await driver.close()


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

    # Do the thing!
    _ = asyncio.run(run_and_clean(driver, LAST_FM_API_KEY))
