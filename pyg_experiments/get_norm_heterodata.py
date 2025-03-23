#!/usr/bin/env python3


import torch
from torch_geometric.data import HeteroData


def get_norm_tensor(tensor: torch.Tensor, m: float, sd: float) -> torch.Tensor:
    return (tensor - m) / sd


def get_m_sd(tensor: torch.Tensor) -> tuple[float, float]:
    m = float(torch.mean(tensor))
    sd = float(torch.std(tensor))
    return m, sd


def main(cut_year, cut_month, percentile):
    train_hd: HeteroData = torch.load(f"{data_dir}train_hd_nomatch_{cut_year}_{cut_month}_{percentile}.pt")
    test_hd: HeteroData = torch.load(f"{data_dir}full_hd_nomatch_{percentile}.pt")

    # Artists
    # begin_date
    m, sd = get_m_sd(train_hd["artist"].x[:, 0])
    train_hd["artist"].x[:, 0] = get_norm_tensor(train_hd["artist"].x[:, 0], m, sd)
    test_hd["artist"].x[:, 0] = get_norm_tensor(test_hd["artist"].x[:, 0], m, sd)
    # end_date
    m, sd = get_m_sd(train_hd["artist"].x[:, 1])
    train_hd["artist"].x[:, 1] = get_norm_tensor(train_hd["artist"].x[:, 1], m, sd)
    test_hd["artist"].x[:, 1] = get_norm_tensor(test_hd["artist"].x[:, 1], m, sd)

    # Tracks
    # year
    m, sd = get_m_sd(train_hd["track"].x[:, 1])
    train_hd["track"].x[:, 1] = get_norm_tensor(train_hd["track"].x[:, 1], m, sd)
    test_hd["track"].x[:, 1] = get_norm_tensor(test_hd["track"].x[:, 1], m, sd)

    torch.save(train_hd, f"{data_dir}train_normhd_nomatch_{cut_year}_{cut_month}_{percentile}.pt")
    torch.save(test_hd, f"{data_dir}test_normhd_nomatch_{cut_year}_{cut_month}_{percentile}.pt")


if __name__ == '__main__':

    data_dir = "pyg_experiments/ds/"

    for p in [0, 0.5, 0.75, 0.9]:
        for y in [2019, 2021, 2023]:
            print(y, p)
            main(y, 11, p)
