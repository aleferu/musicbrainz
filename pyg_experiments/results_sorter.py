#!/usr/bin/env python3


import csv


CSV_FILE = "pyg_experiments/results.csv"
SORTED_CSV_FILE = "pyg_experiments/sorted_results.csv"


def read_and_sort_csv():
    with open(CSV_FILE, mode="r", newline="") as file:
        reader = csv.DictReader(file)
        data = list(reader)

    # Convert necessary fields to correct types for sorting
    for row in data:
        row["year"] = int(row["year"])
        row["month"] = int(row["month"])
        row["perc"] = float(row["perc"])
        row["epoch"] = int(row["epoch"])

    # Sort by model, then year, then perc, then epoch
    data.sort(key=lambda x: (x["model"], x["year"], x["perc"], x["epoch"]))

    # Save sorted data
    with open(SORTED_CSV_FILE, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"Sorted data saved to {SORTED_CSV_FILE}")


if __name__ == "__main__":
    read_and_sort_csv()
