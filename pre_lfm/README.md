# `pre_lfm`

Tests and scripts that prepare the data obtained in `mb_extraction` to be merged with LFM and saved into a DB.

- `add_id_final_tracks_csv.py`: Adds an id column to `data/tracks_no_va_merged.csv` and modifies the data so that Neo4j won't complain.
    - `data/tracks_no_va_merged_id.csv`
- `lastfm.ipynb`: Shows how we can use [last.fm](https://www.last.fm/)'s API and extract the information that holds.
- `tags.ipynb`: Extracts MB's tags into a useful format for later use.
    - `data/tags.csv`
- `clean_tags.py`: Using the information stored at the `util/` directory, this script filters the previously generated `tags.csv` and adapts the other CSVs.
    - `data/tags_clean.csv`
    - `data/artist_tags_clean.csv`
    - `data/tracks_no_va_merged_id_clean.csv`
