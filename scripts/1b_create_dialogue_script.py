import numpy as np
import pandas as pd

THRESHOLD = 0.3

if __name__ == "__main__":
    data = pd.read_csv("data/sperm-whale-dialogues-codas-manhattan.csv")
    data["Start"] = data["TsTo"]
    data["End"] = data["TsTo"].values + data["Duration"].values

    rec_counter, item_position = 0, 0
    new_rows = []
    for rec, rec_data in data.groupby("REC"):
        rec_data = rec_data.sort_values("TsTo").reset_index(drop=True)
        ts_to_baseline = rec_data.iloc[0, :]["TsTo"]
        for i, row in rec_data.iterrows():
            if row["Coda"] not in [100, -1]:
                if (i + 1) != rec_data.shape[0]:
                    ornamentation = int(rec_data["Coda"].values[i + 1] == 100)
                else:
                    ornamentation = int(False)

                synchrony_backwards = (
                    (i != 0)
                    and ((row["TsTo"] - rec_data.iloc[i - 1, :]["TsTo"]) < THRESHOLD)
                    and (row["Whale"] != rec_data.iloc[i - 1, :]["Whale"])
                )
                synchrony_forwards = (
                    ((i + 1) < rec_data.shape[0])
                    and ((row["TsTo"] - rec_data.iloc[i + 1, :]["TsTo"]) < THRESHOLD)
                    and (row["Whale"] != rec_data.iloc[i + 1, :]["Whale"])
                )
                synchrony = int(synchrony_backwards or synchrony_forwards)

                row = (
                    rec_counter,
                    item_position,
                    row["Whale"],
                    row["Coda"],
                    ornamentation,
                    synchrony,
                    row["Duration"],
                    row["TsTo"] - ts_to_baseline,
                )
                new_rows.append(row)
                item_position += 1
        rec_counter += 1
        item_position = 0

    dialogue = pd.DataFrame(
        data=new_rows,
        columns=[
            "sequenceId",
            "itemPosition",
            "Whale",
            "Coda",
            "Ornamentation",
            "Synchrony",
            "Duration",
            "TsToAbs",
        ],
    )
    dialogue.to_csv("data/whale-dialogue-script.csv", index=False)
