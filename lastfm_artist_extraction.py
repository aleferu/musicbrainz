#!/usr/bin/env python3


import urllib.parse
import logging
import requests_async as requests
from typing import Any, LiteralString
from neo4j import AsyncDriver, AsyncGraphDatabase, basic_auth
from neo4j.exceptions import TransientError
from dotenv import load_dotenv
import os
import asyncio
import json
import pandas as pd
import re


def get_clean_name(name: str) -> str:
    good_chars = r"[^a-zA-Z0-9 ]"
    return re.sub(good_chars, r"\\\g<0>", name)


async def get_artist_ids_from_names(driver: AsyncDriver, artist_names: list[str]) -> list[dict[str, str]]:
    # Error avoidance
    artist_names_pairs = list(map(lambda name: [name, get_clean_name(name)], filter(lambda name: len(name) > 0, artist_names)))

    query = """
        UNWIND $artist_names AS artist_name
        CALL (artist_name) {
            WITH artist_name
            WHERE NOT artist_name[1] STARTS WITH "NOT" AND
                NOT artist_name[1] STARTS WITH "AND" AND
                NOT artist_name[1] STARTS WITH "OR"
            CALL db.index.fulltext.queryNodes('artist_names_index', artist_name[1]) YIELD node, score
            WHERE score > 1
            RETURN node.main_id AS main_id
            ORDER BY score DESC
            LIMIT 1
        }
        RETURN artist_name[0] AS artist_name, main_id
        ;
    """
    params = {
        "artist_names": artist_names_pairs
    }
    result = await execute_query_return(driver, query, params)

    return result


def get_tag_ids(tags: list[str], tag_mapping: dict[str, set[str]]) -> list[str]:
    result = set()
    for subgenre, ids in tag_mapping.items():
        for tag in tags:
            if subgenre in tag:
                result = result.union(ids)
                break
    result = list(result)
    return result


async def update_artist(driver: AsyncDriver, main_id: str, listeners: int, playcount: int, similar_artists: dict[str, float], tags: list[str], tag_mapping: dict[str, set[str]]):
    # Update tags
    tag_ids = get_tag_ids(tags, tag_mapping)
    if len(tag_ids) > 0:
        query = f"""
            MATCH (a:Artist {{main_id: \"{main_id}\"}})
            UNWIND {str(tag_ids)} AS tag_id
            MATCH (t:Tag {{id: tag_id}})
            MERGE (t)-[:TAGS]->(a)
            MERGE (a)-[:HAS_TAG]->(t)
        """
        _ = await execute_query(driver, query)

    # Update links between artists
    id_match_pairs = list()
    artists_info = await get_artist_ids_from_names(driver, list(similar_artists.keys()))
    for artist_info in artists_info:
        name = artist_info["artist_name"]
        id = artist_info["main_id"]
        match = similar_artists[name]
        id_match_pairs.append((id, match))

    if len(id_match_pairs) > 0:
        query = """
            MATCH (a0:Artist {main_id: $main_id})
            UNWIND $id_match_pairs AS pair
            MATCH (a1:Artist {main_id: pair[0]})
            WHERE pair[0] <> $main_id
            MERGE (a0)-[l0:LAST_FM_MATCH]->(a1)
                ON CREATE SET l0.weight = pair[1]
                ON MATCH SET l0.weight = CASE WHEN l0.weight < pair[1] THEN pair[1] ELSE l0.weight END
            MERGE (a1)-[l1:LAST_FM_MATCH]->(a0)
                ON CREATE SET l1.weight = pair[1]
                ON MATCH SET l1.weight = CASE WHEN l1.weight < pair[1] THEN pair[1] ELSE l1.weight END
            ;
        """
        params = {
            "main_id": main_id,
            "id_match_pairs": id_match_pairs
        }
        _ = await execute_query(driver, query, params)

    # Update individual stats
    query = f"""
        MATCH (a:Artist {{main_id: \"{main_id}\"}})
        SET
            a.listeners = {listeners},
            a.playcount = {playcount},
            a.last_fm_call = true,
            a.in_last_fm = true
        ;
    """
    _ = await execute_query(driver, query)


async def get_artist_info(artist_name: str, last_fm_api_key: str) -> tuple[bool, dict[str, Any] | None]:
    artist_name_clean = urllib.parse.quote(artist_name)
    request_url = f"http://ws.audioscrobbler.com/2.0/?method=artist.getInfo&artist={artist_name_clean}&autocorrect=1&api_key={last_fm_api_key}&format=json"
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

    similar = await requests.get(f"http://ws.audioscrobbler.com/2.0/?method=artist.getSimilar&artist={artist_name_clean}&autocorrect=1&api_key={last_fm_api_key}&format=json")
    similar = similar.json()
    if "similarartists" in similar:
        artist_info["similar"] = similar["similarartists"]["artist"]

    top_tags = await requests.get(f"http://ws.audioscrobbler.com/2.0/?method=artist.getTopTags&artist={artist_name_clean}&autocorrect=1&api_key={last_fm_api_key}&format=json")
    top_tags = top_tags.json()
    if "toptags" in top_tags:
        top_tags["toptags"]["tag"] = list(filter(lambda t: t["count"] >= 5, top_tags["toptags"]["tag"]))  # 5 seems ok
        artist_info["tags"] = top_tags["toptags"]

    return True, artist_info


async def get_artists_from_db(driver: AsyncDriver, artist_count: int) -> list[dict[str, Any]]:
    query = f"MATCH (n: Artist {{last_fm_call: false}}) RETURN n LIMIT {artist_count};"
    query_result = await execute_query_return(driver, query)
    return [r["n"] for r in query_result]


async def execute_query_return(driver: AsyncDriver, query: LiteralString | str, params: None | dict[str, Any] = None) -> list[dict[str, Any]]:
    try:
        async with driver.session() as session:
            logging.info(f"Querying '{query}'...")
            if params is None:
                result = await session.run(query)  # type: ignore
            else:
                result = await session.run(query, params)  # type: ignore
            return await result.data()
    except TransientError as _:
        return await execute_query_return(driver, query, params)
    except Exception as e:
        raise e


async def execute_query(driver: AsyncDriver, query: LiteralString | str, params: None | dict[str, Any] = None) -> None:
    try:
        async with driver.session() as session:
            logging.info(f"Querying '{query}'...")
            if params is None:
                _ = await session.run(query)   # type: ignore
            else:
                _ = await session.run(query, params)  # type: ignore
    except TransientError as _:
        _ = await execute_query(driver, query, params)
    except Exception as e:
        raise e


async def process_artist(driver: AsyncDriver, artist: dict[str, Any], last_fm_api_key: str, tag_mapping: dict[str, set[str]]):
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
        _ = await update_artist(driver, main_id, listeners, playcount, similar_artists, tag_list, tag_mapping)

    # NOT IN API -> no data could be extracted
    else:
        logging.error(f"  Seems like we couldn't extract info for artist {main_id}...")
        query = f"MATCH (a:Artist {{main_id:  \"{main_id}\"}}) SET a.last_fm_call = true, a.in_last_fm = false"
        _ = await execute_query(driver, query)


def get_tag_mapping() -> dict[str, set[str]] | None:
    genres = pd.read_csv("tags_clean.csv", dtype=str)
    genres = {info["genre"]: info["id"] for _, info in genres.iterrows()}
    with open("util/genres_taxonomy.json", "r") as f:
        taxonomy = json.load(f)

    # Error checking
    genre_set = set(genres)
    taxonomy_set = set(taxonomy)
    if not (genre_set.issubset(taxonomy_set) and taxonomy_set.issubset(genre_set)):
        logging.error("Taxonomy and genre information is not synchronized.")
        return None

    name_mapping = dict()
    for main, subs in taxonomy.items():
        for sub in subs:
            if sub in name_mapping:
                name_mapping[sub].append(genres[main])
            else:
                name_mapping[sub] = [genres[main]]

    return dict((genre_name, set(ids)) for genre_name, ids in name_mapping.items())


async def main(driver: AsyncDriver, last_fm_api_key: str):
    tag_mapping = get_tag_mapping()

    if tag_mapping is None:
        return

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
            logging.error("An error has occured.")
            logging.error(e)
            logging.info("Exiting...")
            return

        if artist_count == 0:
            logging.info("Exiting...")
            return

        # Process each artist
        _ = await asyncio.gather(*[process_artist(driver, artist, last_fm_api_key, tag_mapping.copy()) for artist in await get_artists_from_db(driver, artist_count)])


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
