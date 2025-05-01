#!/usr/bin/env python3


import pandas as pd
import matplotlib.pyplot as plt
import os


# Function to create and save the plots
def create_plots_for_combination(model, year, perc):
    # Filter the DataFrame for the current combination
    subset = df[(df["model"] == model) & (df["year"] == year) & (df["perc"] == perc)]

    # Create the necessary directory structure
    dir_path = f"curves/{model}/{year}-{perc}"
    os.makedirs(f"{dir_path}", exist_ok=True)
    os.makedirs(f"{dir_path}", exist_ok=True)

    # Plot for losses
    plt.figure(figsize=(10, 6))
    plt.plot(subset["epoch"], subset["train_loss"], label="Train Loss")
    plt.plot(subset["epoch"], subset["val_loss"], label="Validation Loss")

    # Red dashed line at the 6th last epoch
    sixth_last_epoch = subset["epoch"].iloc[-6]
    plt.axvline(x=sixth_last_epoch, color="red", linestyle="--", label=f"Best epoch ({sixth_last_epoch})")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Losses - Model: {model}, Year: {year}, Perc: {int(100 * (1 - perc))}")
    plt.legend()

    # Save the loss plot
    plt.savefig(f"{dir_path}/loss_plot.png")
    plt.close()

    # Plot for accuracy and F1 score
    plt.figure(figsize=(10, 6))
    plt.plot(subset["epoch"], subset["acc"], label="Accuracy")
    plt.plot(subset["epoch"], subset["f1"], label="F1 Score")

    # Horizontal dashed line at the 6th last F1 value
    sixth_last_f1 = subset["f1"].iloc[-6]
    plt.axhline(y=sixth_last_f1, color="blue", linestyle="--", label=f"Best F1 ({sixth_last_f1:.2f})")

    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title(f"Accuracy & F1 - Model: {model}, Year: {year}, Perc: {int(100 * (1 - perc))}")
    plt.legend()

    # Save the metrics plot
    plt.savefig(f"{dir_path}/metrics_plot.png")
    plt.close()


if __name__ == "__main__":
    # Read the CSV file into a DataFrame
    df = pd.read_csv("pyg_experiments/sorted_results.csv")

    # Get the unique combinations of model, year, and perc
    model_year_perc_combinations = df[["model", "year", "perc"]].drop_duplicates()

    # Iterate through each combination and generate the plots
    for _, row in model_year_perc_combinations.iterrows():
        model = row["model"]
        year = row["year"]
        perc = row["perc"]

        # Create the plots for this combination
        print(model, year, perc)
        create_plots_for_combination(model, year, perc)

    print("Plots generated and saved.")
