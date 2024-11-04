#!/usr/bin/env python3

import logging
from neo4j import AsyncDriver, AsyncGraphDatabase, basic_auth
from dotenv import load_dotenv
import os
import asyncio
import time
import argparse

from lastfm_artist_extraction import get_tag_mapping
from lastfm_track_extraction import get_tracks_from_db, process_track


async def process_tracks_continuously(driver: AsyncDriver, lfm_keys: list[str], x_tracks: int, duration_seconds: int):
    tag_mapping = get_tag_mapping()

    if tag_mapping is None:
        logging.error("Tag mapping is None, exiting.")
        return

    start_time = time.time()

    while time.time() - start_time < duration_seconds:
        # Fetch x tracks from the db
        tracks = await get_tracks_from_db(driver, x_tracks)

        # If no tracks are found, exit the loop
        if not tracks or len(tracks) == 0:
            logging.info("No more tracks to process. Exiting...")
            break

        # Process each track
        _ = await asyncio.gather(
            *[process_track(driver, track, lfm_keys[i % len(lfm_keys)], tag_mapping.copy()) for i, track in enumerate(tracks)]
        )

        elapsed_time = time.time() - start_time
        logging.info(f"Elapsed time: {elapsed_time:.2f} seconds")

    logging.info("Time limit reached. Exiting...")


async def run_and_clean(driver: AsyncDriver, lfm_keys: list[str], x_tracks: int, duration_seconds: int):
    _ = await process_tracks_continuously(driver, lfm_keys, x_tracks, duration_seconds)
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

    default_track_amount = 1
    parser.add_argument(
        "--track_amount",
        type=int,
        default=1,
        help=f"Amount of tracks fetched by the script per iteration. Default: {default_track_amount}.")

    default_key_amount = 1
    parser.add_argument(
        "--key_amount",
        type=int,
        default=1,
        help=f"Amount of keys used by the script. Default: {default_key_amount}.")

    args = parser.parse_args()
    run_time = args.run_time
    assert run_time > 0
    track_amount = args.track_amount
    assert track_amount > 0
    key_amount = args.key_amount
    assert key_amount > 0

    # .env read
    DB_HOST = os.getenv("NEO4J_HOST")
    DB_PORT = os.getenv("NEO4J_PORT")
    DB_USER = os.getenv("NEO4J_USER")
    DB_PASS = os.getenv("NEO4J_PASS")

    lfm_keys = [os.getenv(f"LAST_FM_API_KEY_{i}") for i in range(key_amount)]

    # .env validation
    assert DB_HOST is not None and \
        DB_PORT is not None and \
        DB_USER is not None and \
        DB_PASS is not None and \
        all(key is not None for key in lfm_keys), \
        "INVALID .env"

    # db connection
    driver = AsyncGraphDatabase.driver(f"bolt://{DB_HOST}:{DB_PORT}", auth=basic_auth(DB_USER, DB_PASS))

    # Do the thing!
    _ = asyncio.run(run_and_clean(driver, lfm_keys, track_amount, run_time))  # type: ignore
