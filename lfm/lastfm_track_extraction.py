#!/usr/bin/env python3


import urllib.parse
import logging
from typing import Any
from neo4j import AsyncDriver, AsyncGraphDatabase, basic_auth
from dotenv import load_dotenv
import os
import asyncio

from lastfm_artist_extraction import get_tag_mapping, get_tag_ids, execute_query, execute_query_return, make_request



async def update_track(driver: AsyncDriver, track_id: str, listeners: int, playcount: int, tags: list[str], tag_mapping: dict[str, set[str]]):
    # Update tags
    tag_ids = get_tag_ids(tags, tag_mapping)
    if len(tag_ids) > 0:
        query = f"""
            MATCH (r:Track {{id: \"{track_id}\"}})
            UNWIND {str(tag_ids)} AS tag_id
            MATCH (t:Tag {{id: tag_id}})
            MERGE (t)-[:TAGS]->(r)
            MERGE (r)-[:HAS_TAG]->(t)
        """
        _ = await execute_query(driver, query)

    # Update individual stats
    query = f"""
        MATCH (r:Track {{id:  \"{track_id}\"}})
        SET
            r.listeners = {listeners},
            r.playcount = {playcount},
            r.last_fm_call = true,
            r.in_last_fm = true
        ;
    """
    _ = await execute_query(driver, query)


async def get_track_info(track_name: str, artist_name: str, last_fm_api_key: str) -> tuple[bool, dict[str, Any] | None]:
    track_name_clean = urllib.parse.quote(track_name)
    artist_name_clean = urllib.parse.quote(artist_name)

    info = await make_request(f"http://ws.audioscrobbler.com/2.0/?method=track.getInfo&artist={artist_name_clean}&track={track_name_clean}&autocorrect=1&api_key={last_fm_api_key}&format=json")

    if info is None:
        return True, None
    if "track" not in info:
        logging.critical("Track not found in request!")
        return False, None

    track_info = info["track"]

    top_tags = await make_request(f"http://ws.audioscrobbler.com/2.0/?method=track.getTopTags&track={track_name_clean}&artist={artist_name_clean}&autocorrect=1&api_key={last_fm_api_key}&format=json")
    if top_tags is not None and "toptags" in top_tags:
        top_tags["toptags"]["tag"] = list(filter(lambda t: t["count"] >= 5, top_tags["toptags"]["tag"]))  # 5 seems ok
        track_info["tags"] = top_tags["toptags"]

    return True, track_info


async def get_artists_from_track(driver: AsyncDriver, track_id: str) -> list[str]:
    query = f"MATCH (r:Track {{id: \"{track_id}\"}})-[]->(a:Artist) UNWIND(a.known_names) AS names RETURN COLLECT(names) AS names;"
    result = await execute_query_return(driver, query)
    return result[0]["names"]


async def get_tracks_from_db(driver: AsyncDriver, track_count: int) -> list[dict[str, Any]]:
    query = f"MATCH (n:Track {{last_fm_call: false}}) WHERE n.name IS NOT NULL RETURN n LIMIT {track_count};"
    query_result = await execute_query_return(driver, query)
    return [r["n"] for r in query_result]


async def process_track(driver: AsyncDriver, track: dict[str, Any], last_fm_api_key: str, tag_mapping: dict[str, set[str]]):
    track_id = track["id"]
    track_name = track["name"]
    logging.info(f"FOUND Track WITH id '{track_id}' and name '{track_name}'")

    artists = await get_artists_from_track(driver, track_id)
    logging.info("Looping through its artists...")

    in_db = False
    listeners = 0
    playcount = 0
    tags: set[str] = set()

    for artist_name in artists:
        logging.info(f"  Trying with the artist name {artist_name}")
        success, track_info = await get_track_info(track_name, artist_name, last_fm_api_key)

        if not success:
            logging.critical("UNEXPECTED ERROR!")
            exit(1)

        if success and track_info is None:
            in_db = in_db or False
            continue

        assert track_info is not None, "LSP was complaining"

        in_db = True

        listeners += int(track_info.get("listeners", "0"))
        logging.info(f"    Listeners: {listeners}")
        playcount += int(track_info.get("playcount", "0"))
        logging.info(f"    Playcount: {playcount}")

        if (track_tags := track_info.get("toptags", False)):
            if (track_tags := track_tags.get("tag", False)):
                tag_list = list(tag["name"] for tag in track_tags)
                tags.update(tag_list)
                logging.info(f"    Found tags: {tag_list}")

    if in_db:
        tag_list = list(tags)
        logging.info(f"Updating track {track_id} with {listeners} listeners, {playcount} playcount, and {len(tag_list)} tags...")
        _ = await update_track(driver, track_id, listeners, playcount, tag_list, tag_mapping)

    else:
        logging.error(f"  Seems like we couldn't extract info for track {track_id}...")
        query = f"MATCH (r:Track {{id: \"{track_id}\"}}) SET r.last_fm_call = true, r.in_last_fm = false"
        _ = await execute_query(driver, query)


async def main(driver: AsyncDriver, last_fm_api_key: str):
    tag_mapping = get_tag_mapping()

    if tag_mapping is None:
        return

    # Ctrl + c for exit
    # Or track_count as 0
    while True:
        try:
            # Get a few tracks from the Neo4j's DB
            track_count = int(input("How many tracks do you want to query? "))

            if track_count < 0:
                raise ValueError("Please select a valid number of tracks. Got a negative number.")

        # Error handling (or not handling)
        except ValueError as e:
            logging.info(e)
            continue
        except Exception as e:
            logging.error("An error has occured.")
            logging.error(e)
            logging.info("Exiting...")
            return

        if track_count == 0:
            logging.info("Exiting...")
            return

        # Process each artist
        _ = await asyncio.gather(*[process_track(driver, track, last_fm_api_key, tag_mapping.copy()) for track in await get_tracks_from_db(driver, track_count)])


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
    LAST_FM_API_KEY = os.getenv("LAST_FM_API_KEY_0")

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
