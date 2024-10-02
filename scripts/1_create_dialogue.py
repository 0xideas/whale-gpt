"""
This code processes whale vocalization data to generate a structured **dialogue** format from the previously processed **coda sequences**. Here’s a breakdown of what each section of the code does:

**Special coda values**:
 - 98: silence
 - 99: change between vocalizations
 - 100: whether the next row contains ornamentation

### 1. **Constants and Data Loading**:
- **THRESHOLD = 0.3**: This constant is used to define the maximum time gap (in seconds) between two codas for them to be considered simultaneous.
- The script reads the file `sperm-whale-dialogues-codas-manhattan.csv` into a pandas DataFrame called `data`.
  - The `Start` and `End` columns are computed based on the `TsTo` (timestamp) and `Duration` of each coda.

### 2. **Organizing Data by Recordings**:
- **Grouping by REC**: The script processes each whale recording (`REC`) individually using `groupby`.
- It loops through each whale in a recording (`Whale`) and identifies:
  - Whether the next row contains **ornamentation** (a Coda ID of 100).
  - Whether the whale in the current row is the **primary whale** in the recording.
  - For each coda, it stores this information into a dictionary (`vals`) with fields: `REC`, `Coda`, `Ornamentation`, `Duration`, `TsTo`, and `PrimaryWhale`.

### 3. **Handling Vocalization Changes**:
- **Vocalization Change Detection**: It checks if the vocalization has changed from the previous row. If it detects a change, it calculates the **duration between vocalizations** and adds a row with a special Coda ID of `99` to signify the change between vocalizations.

### 4. **Building New Dialogue Rows**:
- The script iterates over `new_vals`, which contains information for each coda and its associated details (such as `Coda`, `Ornamentation`, `Duration`, `PrimaryWhale`).
- It processes codas in pairs based on their timestamps:
  - **Primary Whale Handling**: If the coda belongs to the primary whale, it handles the encoding accordingly.
  - **Parallel Condition**: It checks if two whales are vocalizing at nearly the same time (within the `THRESHOLD` of 0.3 seconds). If so, it encodes the codas from both whales into the same row.
  - **Skipping Rows**: If a pair of codas from different whales are processed together, the next iteration skips the second whale's row to avoid duplicate entries.

- The new rows are stored in `new_rows`, and each row contains:
  - `REC`: The recording ID.
  - `item_position`: The position in the sequence of the dialogue.
  - `Coda1`, `Ornamentation1`, `Duration1`: The primary whale’s data.
  - `Coda2`, `Ornamentation2`, `Duration2`: The other whale’s data (if applicable).

### 5. **Filtering and Final Processing**:
- The script filters up to 10 prior rows from `new_rows` to remove PrimaryWhale rows with `Coda1` values of `98` (indicating silence). 
  - This restricts the data to rows where the PrimaryWhale is "speaking" or "speaking" soon
- It checks previous rows (within a window of 10 rows) to ensure all relevant entries are included and adds them to `new_rows_filtered`.

### 6. **Saving the Dialogue Data**:
- After filtering, the processed dialogue data is stored in a new DataFrame `dialogue` with columns:
  - `sequenceId`: The recording sequence ID.
  - `itemPosition`: The position in the dialogue sequence.
  - `Coda1`, `Ornamentation1`, `Duration1`: Data for the primary whale.
  - `Coda2`, `Ornamentation2`, `Duration2`: Data for the other whale (if applicable).

- The final dialogue data is saved to `data/whale-dialogues.csv`.

### **Purpose of the Script**:
This script takes the processed whale coda sequences and generates a **dialogue format** that represents the vocal interactions between multiple whales. It identifies moments where whales vocalize simultaneously and encodes those into a single row. This output can then be used for further analysis or model training to study the structure and patterns of whale communications.
"""
import numpy as np
import pandas as pd

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
                    if (i + 1) != rec_data.shape[0]:
                        ornamentation = int(rec_data["Coda"].values[i + 1] == 100)
                    else:
                        ornamentation = int(False)

                    primary_whale = whale == row["Whale"]
                    vals = {
                        "REC": rec_counter,
                        "Coda": row["Coda"],
                        "Ornamentation": ornamentation,
                        "Duration": row["Duration"],
                        "TsTo": row["TsTo"],
                        "PrimaryWhale": primary_whale,
                    }
                    new_vals.append(vals)

                    vocalization_change = (i != 0) and (
                        rec_data["Vocalization"].values[i - 1] != row["Vocalization"]
                    )
                    if vocalization_change:
                        start_moment = (
                            rec_data[["TsTo", "Duration"]].values[i - 1, :].sum()
                        )
                        duration = max(0.0, row["TsTo"] - start_moment)
                        change_vals = {
                            "REC": rec_counter,
                            "Coda": 99,
                            "Ornamentation": 0,
                            "Duration": duration,
                            "TsTo": start_moment,
                            "PrimaryWhale": primary_whale,
                        }

        rec_counter += 1

    new_rows = []
    skip_next = False
    item_position = 0
    for i, vals in enumerate(new_vals):
        if not skip_next:
            vals2 = (vals["Coda"], vals["Ornamentation"], vals["Duration"])
            none_vals = (98, 0, 0.0)

            if (i + 1) != len(new_vals):
                next_vals = new_vals[i + 1]
                next_vals2 = (
                    next_vals["Coda"],
                    next_vals["Ornamentation"],
                    next_vals["Duration"],
                )
                if (
                    (vals["PrimaryWhale"] != next_vals["PrimaryWhale"])
                    and (vals["REC"] == next_vals["REC"])
                    and (next_vals["TsTo"] - vals["TsTo"]) < THRESHOLD
                ):
                    parallel_condition = True
                else:
                    parallel_condition = False
            else:
                parallel_condition = False

            if vals["Coda"] == 99:
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

            if (i + 1) != len(new_vals):
                next_vals = new_vals[i + 1]
                if vals["REC"] != next_vals["REC"]:
                    item_position = 0
        else:
            skip_next = False

    new_rows_filtered = []
    i_in_rows_filtered = set()
    print(f"{len(new_rows) = }")
    for i, row in enumerate(new_rows):
        if row[2] != 98:
            j_start = max(0, i - 10)
            for j, row2 in enumerate(new_rows[j_start:i]):
                jj = j_start + j
                if jj not in i_in_rows_filtered and row[0] == row2[0]:
                    new_rows_filtered.append(row2)
                    i_in_rows_filtered.add(jj)

            new_rows_filtered.append(row)
            i_in_rows_filtered.add(i)
    print(f"{len(new_rows_filtered) = }")

    dialogue = pd.DataFrame(
        data=new_rows_filtered,
        columns=[
            "sequenceId",
            "itemPosition",
            "Coda1",
            "Ornamentation1",
            "Duration1",
            "Coda2",
            "Ornamentation2",
            "Duration2",
        ],
    )
    dialogue.to_csv("data/whale-dialogues.csv", index=False)
