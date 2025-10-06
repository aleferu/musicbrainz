# tiktok

**WIP**

The idea is to extract all the information that we can for the artists in our DB. For that, we first look for tiktok accounts in [Wikidata](https://www.wikidata.org/wiki/Wikidata:Main_Page) and in Musicbrainz and then we do the API calls.

- `get_data.sh` gets a JSONL with information about all entities inside Wikidata with a tiktok account associated.
- `update_mb_ids.py` is in charge of updating our artist nodes with their GIDs (only those with a tiktok account associated in Wikidata).
- `mb_tiktok_mapping.ipynb` gets the mapping for artist-tiktokusername stored in Musicbrainz.

## WIP
