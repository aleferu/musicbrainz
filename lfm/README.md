# `lfm`

Scripts that get the data from LFM and imports everything into neo4j.

- `neo4j_import.py`: Imports our current dataset into a [Neo4j database](https://neo4j.com/).
    - From now on the idea is to never manage several files anymore and keep it in this Neo4j DB.
- `lastfm_*_extraction*.py`: Imports the information that we can extract from [last.fm](https://www.last.fm/)'s API into the [Neo4j database](https://neo4j.com/). Files used:
    - `lastfm_artist_extraction.py`
    - `lastfm_artist_extraction_cont.py`
    - `lastfm_track_extraction.py`
    - `lastfm_track_extraction_cont.py`
- `add_more_artist_info.py`: Extracts even more information related to artists in Musicbrainz and stores it in Neo4j.
- `pg_weight.py`: Adds a normalized weight field for all edges called `pg_weight`.
