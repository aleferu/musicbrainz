#!/usr/bin/env python3


from flask import Flask, request, jsonify
import csv
import os


app = Flask(__name__)


CSV_FILE = "pyg_experiments/results.csv"
HEADER = ["model", "year", "month", "perc", "epoch", "train_loss", "val_loss", "acc", "prec", "rec", "f1", "auc", "tp", "fp", "fn", "tn", "best_threshold", "done"]


# Ensure the CSV file exists with the correct headers
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(HEADER)


@app.route("/save_results", methods=["POST"])
def save_results():
    data: dict | None = request.json
    assert data is not None

    # Validate incoming data
    if not all(key in data for key in HEADER):
        return jsonify({"error": "Missing fields in request"}), 400

    # Append data to CSV file
    with open(CSV_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([data[key] for key in HEADER])

    return jsonify({"message": "Data saved successfully"}), 200

if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)
