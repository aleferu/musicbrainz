# `mb_extraction`

Scripts and notebooks used for getting the data out of MB.

- `recordings_and_releases.ipynb`: Summary of all I've been learning/working on about `recordings`, `releases` and `tracks`, as well of the needed code to extract CSVs and some graphs.
    - `data/tracks-*.csv`, * being from 1 to 5.
- `genres.ipynb`: Shows that the genre entity is useless in MB at the moment. At least I can't find a way of using it.
- `various_artists_cleanup.py`: Script that merges the tracks CSVs while removing the artist with ID 1 ("Various Artists").
    - `data/tracks_no_va.csv`
- `artist_artist.ipynb`: Shows how to generate CSVs containing information on relationships between artists and how to merge different instances of the same artist. It also exports all the information gathered until this point, but modified.
    - `data/artists.jsonl`
    - `data/relationships.csv`
    - `data/artist_tags.csv`
    - `data/tracks_no_va_merged.csv`
- `relationships_cleanup.py`: Generates a version of the relationships CSV without duplicates.
    - `data/relationships_clean.csv`