# MusicBrainz database study

This repository serves as a way of working with multiple computers and as a way of showing my work to the professor that guides me.

## .env file

A .env file is needed in order to configure the database connection. An example .env file is provided. Modify its contents to satisfy your needs.

## Contents

- `recordings_and_releases.ipynb`: Summary of all I've been learning/working on about `recordings` and `releases`, as well of the needed code to extract CSVs and some graphs. 
- `genres.ipynb`: Shows that genres are impossible to extract via MusicBrainz. At least I can't find the way at the moment.
- `artist_artist.ipynb`: Shows how to generate CSVs containing information on relationships between artists and how to merge different instances of the same artist.
- `various_artists_cleanup.py`: Script that merges the releases CSVs while removing the artist with ID 1 ("Various Artists").

## Dependencies

A working python environment with the following:

```
psycopg2 pandas sqlalchemy matplotlib python-dotenv notebook
```

Use conda, pip... Whatever. Example:

```bash
conda create -n foo psycopg2 pandas sqlalchemy matplotlib python-dotenv notebook
```

## Database installation

I followed [this link](https://musicbrainz.org/doc/MusicBrainz_Server/Setup), section **Setup from source code**. In theory you can install the database without the web server but I didn't want to mess up by skipping a crucial step by accident. The web server installation is very fast, unlike the database installation, so I recommend doing it that way.

## LICENSE

No license right now as I don't know if this work can be exposed to the public at the moment of writing this README.
