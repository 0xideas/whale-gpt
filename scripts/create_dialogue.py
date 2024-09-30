import pandas as pd
import numpy as np


THRESHOLD = 0.3

if __name__ == "__main__":
    data = pd.read_csv("data/sperm-whale-dialogues-codas-manhattan.csv")
    data["Start"] = data["TsTo"]
    data["End"] = data["TsTo"].values + data["Duration"].values

    rec_counter = 0
    new_vals = []
    for rec, rec_data in data.groupby("REC"):
        rec_data = rec_data.sort_values("TsTo").reset_index(drop=True)
        whales = np.unique(rec_data["Whale"].values)
        for whale in whales:

            for i, row in rec_data.iterrows():
                
                if row["Coda"] not in [100, -1]:

                    if (i+1) != rec_data.shape[0]:
                        ornamentation = int(rec_data["Coda"].values[i+1] == 100)
                    else:
                        ornamentation = int(False)

                    primary_whale = whale == row["Whale"]
                    vals = {"REC": rec_counter, "Coda": row["Coda"], "Ornamentation": ornamentation, "Duration": row["Duration"],  "TsTo": row["TsTo"], "PrimaryWhale": primary_whale}
                    new_vals.append(vals)

                    vocalization_change = ((i != 0) and (rec_data["Vocalization"].values[i-1] != row["Vocalization"]))
                    if vocalization_change:
                        start_moment = rec_data[["TsTo", "Duration"]].values[i-1,:].sum()
                        duration = max(0.0, row["TsTo"] - start_moment)
                        change_vals = {"REC": rec_counter, "Coda": 99, "Ornamentation": 0, "Duration": duration, "TsTo": start_moment, "PrimaryWhale": primary_whale }

        rec_counter += 1

    new_rows = []
    skip_next = False
    item_position = 0
    for i, vals in enumerate(new_vals):
        if not skip_next:
            vals2 = (vals["Coda"], vals["Ornamentation"], vals["Duration"])
            none_vals = (98, 0, 0.0)

            if (i+1) != len(new_vals):
                next_vals = new_vals[i+1]
                next_vals2 = (next_vals["Coda"], next_vals["Ornamentation"], next_vals["Duration"])
                if (vals["PrimaryWhale"] != next_vals["PrimaryWhale"]) and (vals["REC"] == next_vals["REC"]) and (next_vals["TsTo"] - vals["TsTo"]) < THRESHOLD:
                    parallel_condition = True
                else:
                    parallel_condition = False
            else:
                parallel_condition = False

            if (vals["Coda"] == 99):
                new_row = (vals["REC"], item_position, *vals2, *vals2)
            elif vals["PrimaryWhale"] and not parallel_condition:
                new_row = (vals["REC"], item_position, *vals2, *none_vals)
            elif not parallel_condition:
                new_row = (vals["REC"], item_position, *none_vals, *vals2)
            else:
                if vals["PrimaryWhale"]:
                    new_row = (vals["REC"], item_position, *vals2, *next_vals2)
                else:
                    new_row = (vals["REC"], item_position, *next_vals2, *vals2)
                skip_next = True
            new_rows.append(new_row)
            item_position += 1

            if (i+1) != len(new_vals):
                next_vals = new_vals[i+1]
                if vals["REC"] != next_vals["REC"]:
                    item_position = 0
        else:
            skip_next = False

    new_rows_filtered = []
    i_in_rows_filtered = set()
    print(f"{len(new_rows) = }")
    for i, row in enumerate(new_rows):
        if row[2] != 98:
            j_start = max(0, i-10)
            for j, row2 in enumerate(new_rows[j_start:i]):
                jj = j_start + j
                if jj not in i_in_rows_filtered and row[0] == row2[0]:
                    new_rows_filtered.append(row2)
                    i_in_rows_filtered.add(jj)
            
            new_rows_filtered.append(row)
            i_in_rows_filtered.add(i)
    print(f"{len(new_rows_filtered) = }")


    dialogue = pd.DataFrame(data=new_rows_filtered, columns=["sequenceId", "itemPosition", "Coda1", "Ornamentation1", "Duration1", "Coda2", "Ornamentation2", "Duration2"])
    dialogue.to_csv("data/whale-dialogues.csv", index=False)



