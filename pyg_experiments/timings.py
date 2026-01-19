#!/usr/bin/env python3


import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


if __name__ == '__main__':
    # Ensure output directory exists
    os.makedirs("timing_imgs", exist_ok=True)

    # Mapping from file to paper model name
    file_to_model = {
        "main_mb_timings.data": "2 Convs+concat+2 Dense",
        "nocat_mb_timings.data": "MUSYNERGY",
        "oneconv_mb_timings.data": "1 Conv + 2 Dense",
        "oneconvoneff_mb_timings.data": "1 Conv + 1 Dense"
    }

    # Load timing data
    timing_data = {}
    for filename, model_name in file_to_model.items():
        with open(filename, "rb") as f:
            timing_data[model_name] = pickle.load(f)[1:]

    # Prepare data for plotting
    model_names = list(timing_data.keys())
    samples = [timing_data[name] for name in model_names]
    means = [np.mean(x) for x in samples]
    stds = [np.std(x) for x in samples]

    # Try a few matplotlib themes for visual diversity
    themes = ['default', 'seaborn-v0_8-darkgrid', 'ggplot']

    for i, style in enumerate(themes):
        plt.style.use(style)

        ## 1. Bar Plot (Mean Inference Time)
        plt.figure(figsize=(8, 5))
        plt.bar(model_names, means, color='skyblue')
        plt.ylabel("Average Inference Time (ms)")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(f"timing_imgs/bar_plot_theme{i+1}.png")
        plt.close()

        ## 2. Bar Plot with Error Bars
        plt.figure(figsize=(8, 5))
        plt.bar(model_names, means, yerr=stds, capsize=5, color='lightgreen', edgecolor='black')
        plt.ylabel("Average Inference Time (ms)")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(f"timing_imgs/bar_plot_with_errors_theme{i+1}.png")
        plt.close()

        ## 3. Violin Plot
        plt.figure(figsize=(8, 5))
        sns.violinplot(data=samples)
        plt.xticks(ticks=range(len(model_names)), labels=model_names, rotation=15)
        plt.ylabel("Inference Time (ms)")
        plt.tight_layout()
        plt.savefig(f"timing_imgs/violin_plot_theme{i+1}.png")
        plt.close()

        ## 4. Boxplot
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=samples)
        plt.xticks(ticks=range(len(model_names)), labels=model_names, rotation=15)
        plt.ylabel("Inference Time (ms)")
        plt.tight_layout()
        plt.savefig(f"timing_imgs/boxplot_theme{i+1}.png")
        plt.close()

    print("All plots saved in the 'timing_imgs/' directory.")
