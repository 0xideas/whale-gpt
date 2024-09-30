import pandas as pd
import numpy as np

if __name__ == "__main__":
    data = pd.read_csv("data/sperm-whale-dialogues-codas-manhattan.csv").reset_index(drop=True)

    new_rows = []
    vocalization_change = 0
    speaker_change = 0
    rec_counter, item_position = 0, 0
    for rec, rec_data in data.groupby("REC"):
        rec_data = rec_data.sort_values("TsTo").reset_index(drop=True)

        for i, row in rec_data.iterrows():
            if row["Coda"] not in [100, -1]:
                if (i+1) != rec_data.shape[0]:
                    ornamentation = int(rec_data["Coda"].values[i+1] == 100)
                else:
                    ornamentation = int(False)

                if (i != 0) and (rec_data["Vocalization"].values[i-1] != row["Vocalization"]):
                    vocalization_change = 1 - vocalization_change

                if (i != 0) and (rec_data["Whale"].values[i-1] != row["Whale"]):
                    speaker_change = 1 - speaker_change

                new_row = [rec_counter, item_position, row["Coda"], ornamentation, row["Duration"], vocalization_change, speaker_change]
                new_rows.append(new_row)
                item_position += 1
        rec_counter += 1
        item_position = 0

    new_data = pd.DataFrame(new_rows, columns = ["sequenceId", "itemPosition", "Coda", "Ornamentation", "Duration", "VocalizationChange", "SpeakerChange"])
    new_data.to_csv("data/whale-sequences.csv", index=False, sep=",", decimal=".", float_format=lambda x: f'{x:.7f}')
    # coda, ornamentation, duration, new vocalization, speaker change
