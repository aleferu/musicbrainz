#!/usr/bin/env python3


import pandas as pd
import numpy as np


if __name__ == '__main__':
    print("Reading CSV...")
    relationships = pd.read_csv('relationships.csv', dtype=str)

    # Drop duplicates
    print("Cleaning up...")
    relationships.drop_duplicates(inplace=True)  # Shouldn't be necessary with the next lines, but just in case

    # Drop duplicates (A,Foo,B,Bar,1 and B,Bar,A,Foo,1 is a duplicate)
    relationships[['id0_sorted', 'id1_sorted']] = pd.DataFrame(np.sort(relationships[['id0', 'id1']], axis=1))
    relationships.drop_duplicates(subset=['id0_sorted', 'id1_sorted', 'relationship_type'], inplace=True)
    relationships.drop(columns=['id0_sorted', 'id1_sorted'], inplace=True)

    # Removing unnecessary rows
    mask = relationships["id0"] == relationships["id1"]
    relationships = relationships[~mask]

    # Export
    print("Exporting...")
    relationships.to_csv("relationships_clean.csv", index=False)

    print("Done!")
