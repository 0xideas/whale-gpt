import json

import numpy as np
import pandas as pd

rhythm = {
    "1+1+3": 5,
    "1+31": 4,
    "1+32": 4,
    "1-NOISE": -1,
    "10-NOISE": -1,
    "10R": 16,
    "10i": 17,
    "2+3": 7,
    "2-NOISE": -1,
    "3-NOISE": -1,
    "3D": 0,
    "3R": 1,
    "4-NOISE": -1,
    "4D": 2,
    "4R1": 3,
    "4R2": 3,
    "5-NOISE": -1,
    "5R1": 6,
    "5R2": 6,
    "5R3": 6,
    "6-NOISE": -1,
    "6R": 8,
    "6i": 9,
    "7-NOISE": -1,
    "7D1": 10,
    "7D2": 10,
    "7R": 10,
    "7i": 11,
    "8-NOISE": -1,
    "8D": 12,
    "8R": 12,
    "8i": 13,
    "9-NOISE": -1,
    "9R": 14,
    "9i": 15,
}


def standardize(dataframe, len):
    sums = np.sum(dataframe.values[:, :len], axis=1).reshape(-1, 1)
    denom = np.concatenate([sums for _ in range(len)], axis=1)
    cumsum = np.cumsum(dataframe.values[:, :len] / denom, axis=1)
    return np.concatenate([np.zeros((cumsum.shape[0], 1)), cumsum], axis=1)


def get_coda_data(path="data/DominicaCodas.csv"):
    codas = pd.read_csv(path)
    codas["CodaTypeConverted"] = [rhythm[v] for v in codas["CodaType"]]
    codas_groups = dict(
        tuple(codas.groupby("CodaTypeConverted")[[f"ICI{i+1}" for i in range(9)]])
    )
    lengths = {
        k: (np.min(np.sum(v.values != 0, 1)), np.max(np.sum(v.values != 0, 1)))
        for k, v in codas_groups.items()
    }
    standardised = {
        k: standardize(v, lengths[k][0]) for k, v in codas_groups.items() if k != -1
    }
    means = {k: v.mean(axis=0) for k, v in standardised.items()}
    means_trimmed = {k: v[1:] for k, v in means.items()}

    return means_trimmed


if __name__ == "__main__":
    dialogues = pd.read_csv("data/sperm-whale-dialogues.csv")
    means = get_coda_data()
    with open("data/coda-means.json", "w") as f:
        f.write(json.dumps({k: list(v) for k, v in means.items()}))
