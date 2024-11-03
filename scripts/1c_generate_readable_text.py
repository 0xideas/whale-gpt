# Written by Morgan Rivers!!
import argparse
import pandas as pd
import numpy as np

"""
This python script generates a dialogue using the whale dialogue script data.
It does so by first labeling the whale by its number ("Whale" column) and the words that it says.
When another whale speaks, the constructedText column is concatenated until another whale speaks or the conversation ends.
New conversations, defined by having a different "sequenceId", are separated and labeled by their sequenceId.
This also includes proper timings: printing the time of the new conversations, as well as reordering the display by
timestamp and printing when codas are spoken effectively simultaneously.
"""
THRESHOLD = 0.3
SILENCE_THRESHOLD = 5.0
# Reconstruct 'ConstructedString'
def return_tempo(dur):
    if dur < 0.45:
        return 0
    elif dur < 0.61:
        return 1
    elif dur < 0.93:
        return 2
    elif dur < 1.08:
        return 3
    else:
        return 4

def get_coda_string_representation(rhythm_index, tempo, ornamentation):
    rhythm_char = chr(ord("a") + rhythm_index)
    if ornamentation == 1:
        rhythm_char = rhythm_char.upper()
    else:
        rhythm_char = rhythm_char.lower()

    constructed_string = rhythm_char + str(tempo + 1)
    return constructed_string

# Function to determine rubato
def determine_rubato(
    word_string_previous, word_string, duration_previous, duration, t_diff
):
    if t_diff > 10:
        return " "

    # make sure the coda is one of the possibilities ('a' through 'r')
    assert word_string[0].lower() in [chr(i) for i in range(ord('a'), ord('r') + 1)]
    assert word_string_previous[0].lower() in [chr(i) for i in range(ord('a'), ord('r') + 1)]

    rhythm_previous = word_string_previous.lower()[0]
    rhythm = word_string.lower()[0]
    tempo_previous = word_string_previous.lower()[1]
    tempo = word_string.lower()[1]

    # Rubato is defined as a change of duration within the same tempo and rhythm class
    if rhythm_previous != rhythm:
        return " "

    if tempo_previous != tempo:
        return " "

    duration_delta = duration - duration_previous

    return duration_delta


# Function to categorize rubato
def categorize_rubato(rubato):
    quantile_25th = -0.02142
    quantile_75th = 0.01846

    # Assign categories based on these quantiles
    if rubato < quantile_25th:
        # Decreasing
        return str("\\")
    elif rubato < quantile_75th:
        # Constant
        return str("-")
    else:
        # Increasing
        return str("/")



# Function to print chorus
def print_chorus(chorus_whales_data, f):
    sorted_keys = sorted(chorus_whales_data)
    sorted_texts = [
        chorus_whales_data[key] for key in sorted_keys
    ]  # Extract values in the sorted order of keys
    chorus_string = f"In chorus, whales {', '.join(map(str, sorted_keys))}: {' '.join(sorted_texts)}."
    f.write(chorus_string + "\n")


# Function to print time without vocalizations
def format_time_no_vocalizations(time_diff):
    if time_diff < 60:
        # Less than a minute
        rounded = 5 * (time_diff // 5)
        unit_label = "second"
    elif time_diff < 3600:
        # Less than an hour
        units = time_diff // 60  # Convert to minutes
        if units < 5:
            rounded = units  # Keep exact if less than 5 minutes
        else:
            rounded = 5 * round(units / 5)
        unit_label = "minute"
    elif time_diff < 86400:
        # Less than a day, but more than an hour
        units = time_diff // 3600  # Convert to hours
        if units < 5:
            rounded = units  # Keep exact if less than 5 hours
        else:
            rounded = 5 * round(units / 5)
        unit_label = "hour"
    else:
        # One day or more
        units = time_diff // 86400  # Convert to days
        if units < 5:
            rounded = units  # Keep exact if less than 5 days
        else:
            rounded = 5 * round(units / 5)
        unit_label = "day"
    unit_label += "" if rounded == 1 else "s"
    return(f"{int(rounded)} {unit_label})")


def group_annotation(annotation: list[dict]):
    annotation_groups = [[]]
    for i, annotat in enumerate(annotation):
        whale_in_annotation_group = max(([False] + [annotat["whale_number"] == ann_gr["whale_number"] for ann_gr in annotation_groups[-1]]))
        if i == 0 or annotat["time_delta"] > THRESHOLD or whale_in_annotation_group:
            annotation_groups.append([annotat])
        else:
            annotation_groups[-1].append(annotat)
    annotation_groups = [sorted(annotation_group, key=lambda x: x["whale_number"]) for annotation_group in annotation_groups]
    return(annotation_groups[1:])

def get_annotation_group_string(annotation_group):
    assert len(annotation_group) >= 1, len(annotation_group)

    if annotation_group[0]["time_delta"] > SILENCE_THRESHOLD:
        time_string = format_time_no_vocalizations(annotation_group[0]["time_delta"])
        silence_string = f"\n(silence for {time_string}\n\n"
    else:
        silence_string = ""

    if len(annotation_group) == 1:
        return(f"{silence_string}Whale {annotation_group[0]['whale_number']}: {annotation_group[0]['text']}")
    else:
        whales_string = "Whales " + ', '.join([str(annotation['whale_number']) for annotation in annotation_group])
        coda_string = ', '.join([annotation['text'] for annotation in annotation_group])
        return(f"{silence_string}{whales_string}: {coda_string}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = vars(parser.parse_args())

    # Load the data from the new data format
    path = args["path"]
    data = pd.read_csv(path)

    # Initialize all the different conversations as empty arrays
    annotations = {sequence_id: [] for sequence_id in data["sequenceId"].unique()}
    # Then we go over the entries of the annotation file one by one (row-wise) and append the annotation to the list corresponding to the sequence_id
    prev_sequence_id = -np.inf
    for i, row in data.iterrows():
        sequence_id = row["sequenceId"]
        time_delta = float(row["TimeDelta"]) 
        if prev_sequence_id != sequence_id:
            prev_rhythm_letter = None
            is_first_in_sequence = True
            prev_coda_duration = -np.inf
        else:
            # this is another coda in the sequence
            is_first_in_sequence = False

        whale_number = int(row["Whale"])
        coda = int(row["Coda"]) 
        ornamentation = int(row["Ornamentation"])
        synchrony = int(row["Synchrony"])
        duration = float(row["Duration"])

        tempo = return_tempo(duration)

        rhythm_letter = get_coda_string_representation(coda, tempo, ornamentation)

        annotations[sequence_id].append(
            {
                "is_first_in_sequence": is_first_in_sequence,
                "whale_number": whale_number,
                "coda": coda,
                "rhythm_letter": rhythm_letter,
                "tempo": tempo,
                "duration": duration,
                "time_delta": np.exp(time_delta) - 0.1,
                "synchrony": synchrony,
                "prev_coda_duration": prev_coda_duration,
                "prev_rhythm_letter": prev_rhythm_letter,
            }
        )

        prev_sequence_id = sequence_id
        prev_coda_duration = duration
        prev_rhythm_letter = rhythm_letter


    # CODA grouping parameters:
    max_diff = 10  # Max time difference otherwise print that there was a pause
    # Initialize an empty list to hold the dialogues
    dialogues = []
    rubato_deltas = []

    # Compose the full text for the coda, including the rubato
    for recording_id, recording in annotations.items():
        for coda_position, this_coda in enumerate(recording):
            if not this_coda["is_first_in_sequence"]:
                rubato = determine_rubato(
                    word_string_previous=this_coda["prev_rhythm_letter"],
                    word_string=this_coda["rhythm_letter"],
                    duration_previous=this_coda["prev_coda_duration"],
                    duration=this_coda["duration"],
                    t_diff=this_coda["time_delta"],
                )
                if rubato != " ":
                    rubato_deltas.append(rubato)
                    rubato_string = categorize_rubato(rubato)
                else:
                    rubato_string = " "
            else:
                rubato_string = " "
            annotations[recording_id][coda_position]["text"] = rubato_string + this_coda["rhythm_letter"]

    # Open the output file
    with open(f"{path.replace('.csv', '-readable.txt')}", "w") as f:
        # Print the dialogues
        for sequence_id, annotation in annotations.items():

            annotation_groups = group_annotation(annotation)

            # Write filename right above the start of the conversation to the file
            f.write(f"Sequence ID: {sequence_id}\n\n")

            for annotation_group in annotation_groups:
                annotation_group_string = get_annotation_group_string(annotation_group)
                f.write(f"{annotation_group_string}\n")

            f.write("\n\n")


