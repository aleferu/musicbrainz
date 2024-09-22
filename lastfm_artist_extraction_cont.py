#!/usr/bin/env python3

import logging
from neo4j import AsyncDriver, AsyncGraphDatabase, basic_auth
from dotenv import load_dotenv
import os
import asyncio
import time
import argparse

from lastfm_artist_extraction import process_artist, get_artists_from_db, get_tag_mapping


async def process_artists_continuously(driver: AsyncDriver, last_fm_api_key: str, x_artists: int, duration_seconds: int):
    tag_mapping = get_tag_mapping()

    if tag_mapping is None:
        logging.error("Tag mapping is None, exiting.")
        return

    start_time = time.time()

    while time.time() - start_time < duration_seconds:
        # Fetch x artists from the db
        artists = await get_artists_from_db(driver, x_artists)

        # If no artists are found, exit the loop
        if not artists or len(artists) == 0:
            logging.info("No more artists to process. Exiting...")
            break

        # Process each artist
        _ = await asyncio.gather(
            *[process_artist(driver, artist, last_fm_api_key, tag_mapping.copy()) for artist in artists]
        )

        elapsed_time = time.time() - start_time
        logging.info(f"Elapsed time: {elapsed_time:.2f} seconds")

    logging.info("Time limit reached. Exiting...")


async def run_and_clean(driver: AsyncDriver, last_fm_api_key: str, x_artists: int, duration_seconds: int):
    _ = await process_artists_continuously(driver, last_fm_api_key, x_artists, duration_seconds)
    _ = await driver.close()


if __name__ == '__main__':
    load_dotenv()

    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Argument parser
    parser = argparse.ArgumentParser(
        description="Fetch LastFM data continously."
    )

    default_run_time = 60
    parser.add_argument(
        "--run_time",
        type=int,
        default=60,
        help=f"Amount of seconds the script will be running (estimate). Default: {default_run_time}."
    )

    default_artist_amount = 1
    parser.add_argument(
        "--artist_amount",
        type=int,
        default=1,
        help=f"Amount of artists fetched by the script per iteration. Default: {default_artist_amount}.")

    args = parser.parse_args()
    run_time = args.run_time
    assert run_time > 0
    artist_amount = args.artist_amount
    assert artist_amount > 0

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
    _ = asyncio.run(run_and_clean(driver, LAST_FM_API_KEY, artist_amount, run_time))
