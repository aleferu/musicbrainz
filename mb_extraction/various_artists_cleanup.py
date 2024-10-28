#!/usr/bin/env python3


import pandas as pd


def get_tracks_df() -> pd.DataFrame:
    # This function must be called after 'recordings_and_releases.ipynb'
    dfs = []
    for i in range(1, 6):
        path = f"./data/tracks-{i}.csv"
        dfs.append(pd.read_csv(path, dtype=str))
    tracks = pd.concat(dfs, ignore_index=True)
    tracks.fillna("", inplace=True)
    return tracks


def remove_various_artist(row: dict[str, str]) -> dict[str, str]:
    # Lower the artist_count
    row["artist_count"] = str(int(row["artist_count"]) - 1)

    # Get the position of the "Various Artists" instance
    various_artist_position = next((i for i in range(5) if row[f"a{i}_id"] == "1"))

    # Remove the instance
    row[f"a{various_artist_position}_id"] = ""
    row[f"a{various_artist_position}_name"] = ""

    # If the "Various Artist" instance was at the end we're finished
    if various_artist_position == row["artist_count"]:
        return row

    # Place another instance at its place, filling the gap
    aux_id = row[f"a{row['artist_count']}_id"]
    aux_name = row[f"a{row['artist_count']}_name"]
    row[f"a{various_artist_position}_id"] = aux_id
    row[f"a{various_artist_position}_name"] = aux_name

    return row


def main() -> None:
    # Data read
    print("Obtaining the data...")
    tracks = get_tracks_df()

    # Only one artist with "Various Artists"
    print("Removing the solo instances of 'Various Artists'")
    tracks = tracks.loc[
        ~((tracks["a0_id"] == "1") & (tracks["artist_count"] == "1"))
    ]

    # The rest
    print("Solving the other instances")
    mask = (tracks[[f"a{j}_id" for j in range(5)]] == "1").any(axis=1)
    tracks.loc[mask] = tracks.loc[mask].apply(remove_various_artist, axis=1)

    # Save the result
    print("Saving the result...")
    tracks.sort_values(by=["artist_count"], inplace=True)
    tracks.to_csv("./data/tracks_no_va.csv", index=False)

    print("DONE!")


if __name__ == '__main__':
    main()
