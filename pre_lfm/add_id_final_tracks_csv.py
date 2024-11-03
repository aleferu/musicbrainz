#!/usr/bin/env python3


import pandas as pd


if __name__ == '__main__':
    print("Reading...")
    data = pd.read_csv("./data/tracks_no_va_merged.csv", dtype=str)

    print("Removing doble quotes...")
    data = data.map(lambda x: x.replace('"', ''), na_action="ignore")

    print("Removing backslashes quotes...")
    data = data.map(lambda x: x.replace('\\', ''), na_action="ignore")

    print("Adding an index column...")
    data["id"] = range(len(data))

    print("Writing...")
    data.to_csv("./data/tracks_no_va_merged_id.csv", index=False)

    print("Done!")
