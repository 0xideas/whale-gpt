"""
This python script generates a dialogue using the whale dialogue script data.
It does so by first labeling the whale by its number ("Whale" column) and the words that it says.
When another whale speaks, the constructedText column is concatenated until another whale speaks or the conversation ends.
New conversations, defined by having a different "sequenceId", are separated and labeled by their sequenceId.

This also includes proper timings: printing the time of the new conversations, as well as reordering the display by
timestamp and printing when codas are spoken effectively simultaneously.

"""

import pandas as pd
import numpy as np

# Load the data from the new data format
data = pd.read_csv("../data/whale-dialogue-script.csv")


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


def get_coda_string_representation(coda, tempo, ornamentation):
    # print("")
    # print("coda")
    # print(coda)
    rhythm_index = coda - 1  # Adjusting to 0-based index
    # print("rhythm_index")
    # print(rhythm_index)
    rhythm_char = chr(ord("a") + rhythm_index)
    # print("rhythm_char")
    # print(rhythm_char)
    # print("ornamentation")
    # print(ornamentation)
    if ornamentation == 1:
        rhythm_char = rhythm_char.upper()
    else:
        rhythm_char = rhythm_char.lower()

    constructed_string = rhythm_char + str(tempo + 1)
    return constructed_string


# Initialize all the different conversations as empty arrays
annotations = {}
for sequence_id in data["sequenceId"].unique():
    annotations[sequence_id] = []


# Then we go over the entries of the annotation file one by one (row-wise) and append the annotation to the list corresponding to the sequence_id
prev_sequence_id = -np.inf
for i, row in data.iterrows():
    sequence_id = row["sequenceId"]
    coda_start_time = float(row["TsToAbs"])  # Start time of the coda
    if prev_sequence_id != sequence_id:
        # this is a new sequence
        delta_from_last_coda = 0
        prev_start_time = 0
        prev_rhythm_letter = None
        is_first_in_sequence = True
        prev_coda_duration = -np.inf
    else:
        # this is another coda in the sequence
        is_first_in_sequence = False
        delta_from_last_coda = coda_start_time - prev_start_time

    whale_number = int(row["Whale"])
    # print("")
    # print("row")
    # print(row)
    assert 100 > int(whale_number) > 0
    coda = int(row["Coda"])  # Assuming this corresponds to 'Rhythm' in old code
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
            "time_delta": delta_from_last_coda,
            "synchrony": synchrony,
            "prev_coda_duration": prev_coda_duration,
            "prev_rhythm_letter": prev_rhythm_letter,
        }
    )

    prev_sequence_id = sequence_id
    prev_start_time = coda_start_time
    prev_coda_duration = duration
    prev_rhythm_letter = rhythm_letter


# CODA grouping parameters:
max_diff = 10  # Max time difference otherwise print that there was a pause


# Function to determine rubato
def determine_rubato(
    word_string_previous, word_string, duration_previous, duration, t_diff
):
    if t_diff > 10:
        return " "

    # make sure the coda is one of the possibilities ('a' through 'r')
    # print("word_string[0].lower()")
    # print(word_string[0].lower())
    # assert word_string[0].lower() in [chr(i) for i in range(ord('a'), ord('r') + 1)]
    # assert word_string_previous[0].lower() in [chr(i) for i in range(ord('a'), ord('r') + 1)]

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


# Initialize an empty list to hold the dialogues
dialogues = []
rubato_deltas = []


# Compose the full text for the coda, including the rubato
for i in range(len(annotations)):
    recording = annotations[i]
    for j in range(len(recording)):
        this_coda = recording[j]
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
        annotations[i][j]["text"] = rubato_string + this_coda["rhythm_letter"]


# Function to print chorus
def print_chorus(chorus_whales_data, f):
    # print("chorus_whales_data")
    # print(chorus_whales_data)
    sorted_keys = sorted(chorus_whales_data)
    sorted_texts = [
        chorus_whales_data[key] for key in sorted_keys
    ]  # Extract values in the sorted order of keys
    chorus_string = f"In chorus, whales {', '.join(map(str, sorted_keys))}: {' '.join(sorted_texts)}."
    f.write(chorus_string + "\n")


# Function to print time without vocalizations
def print_time_no_vocalizations(time_diff, f):
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
    f.write(f"\n(No vocalizations, {int(rounded)} {unit_label})\n\n")


# Open the output file
with open("../data/whale_dialogues.txt", "w") as f:
    # Print the dialogues
    for annotation_id, annotation in annotations.items():
        # Initialize variables
        previous_whale_name = ""  # empty as there is no previous, it's the start of a conversation
        what_last_whale_said = ""
        what_last_whale_said_array = []

        # Write filename right above the start of the conversation to the file
        f.write(f"Sequence ID: {annotation_id}\n")

        # Before starting the dialogue loop
        chorus_whales_data = {}
        previously_in_chorus = False
        previous_whale_utterance = ""

        # Inside the dialogue loop
        for i in range(len(annotation)):
            line = annotation[i]

            # for i in range(len(dialogue["dialogue"])):
            #     line = dialogue["dialogue"][i]

            # Check time difference and manage chorus
            time_diff = line["time_delta"]

            # uncomment below to print the line as well for debugging
            # f.write("\n" + str(line) + f" tdiff: {time_diff}\n")
            if (
                line["synchrony"]
                and previous_whale_name
                and previous_whale_name != line["whale_number"]
            ):
                # if (
                #     line["synchrony"]
                #     and previous_whale_name != line["whale_number"]

                # ):
                # if previous_whale_name and what_last_whale_said: # if not on the first one
                if not previously_in_chorus:
                    # we don't need to repeat this one, it will be in the chorus, so go up to the penultamate entry
                    if len(what_last_whale_said_array) > 1:
                        # If there was any previous stored thing to say, print it.
                        # Also remove the most previous as it's part of the chorus
                        f.write(
                            f"Whale {previous_whale_name}: "
                            + " ".join(what_last_whale_said_array[:-1])
                            + ".\n"
                        )

                # Add current whale to chorus if not already in it
                if previous_whale_name not in chorus_whales_data.keys():
                    chorus_whales_data[previous_whale_name] = (
                        previous_whale_utterance
                    )
                if (
                    line["whale_number"] not in chorus_whales_data.keys()
                    and line["whale_number"] != ""
                ):
                    chorus_whales_data[line["whale_number"]] = line["text"]
                    # if previous_whale_name == "":
                    #     print("Line 297 Whale {previous_whale_name}")
                    #     quit()

                what_last_whale_said = ""
                what_last_whale_said_array = []
                previously_in_chorus = True

            else:
                # Output chorus if it exists and reset
                if (
                    previously_in_chorus
                ):  # if was chorus last time, and not this time.
                    print_chorus(chorus_whales_data, f)
                    chorus_whales_data = {}
                    what_last_whale_said = (
                        f"Whale {line['whale_number']}: {line['text']}"
                    )
                    what_last_whale_said_array.append(line["text"])

                else:
                    # Continue with the regular logic
                    if line["whale_number"] == previous_whale_name:
                        # We know the tdiff was large last time, but that might have been a time that chorus was printed.
                        # we should have loaded the word when first not in chorus into what_last_whale_said_array
                        # we do want to print if the last whale was in chorus before.
                        what_last_whale_said += " " + line["text"]
                        what_last_whale_said_array.append(line["text"])
                    else:  # not in a chorus, not previously in a chorus, and the whale is different.
                        if (
                            previous_whale_name and what_last_whale_said != ""
                        ):  # and, last whale said something, and not first entry
                            f.write(what_last_whale_said + ".\n")

                        what_last_whale_said = (
                            f"Whale {line['whale_number']}: {line['text']}"
                        )
                        what_last_whale_said_array = [line["text"]]
                previously_in_chorus = False

            # we want to split up vocalizations that are a long time apart in text dialogue.
            if (
                time_diff > max_diff
                and not np.isnan(time_diff)
                and not time_diff == np.inf
            ):
                # this cannot be a chorus, as time_diff is high. So it would have printed.
                # Also print past vocalizations of the same whale (which otherwise would be skipped) because its a long pause.
                # we don't want to print this entry though, it needs to be printed after the pause (tdiff is prev - current time)
                if len(what_last_whale_said_array) > 1:
                    f.write(
                        f"Whale {previous_whale_name}: "
                        + " ".join(what_last_whale_said_array[:-1])
                        + ".\n"
                    )
                    # previous_whale_name = ""  # empty as there is no previous, it's the start of a conversation
                    what_last_whale_said = (
                        f"Whale {line['whale_number']}: {line['text']}"
                    )
                    what_last_whale_said_array = [line["text"]]

                print_time_no_vocalizations(time_diff, f)
                time_diff = 0

            previous_whale_name = line["whale_number"]
            previous_whale_utterance = line["text"]

        # After the loop, check if there's an unprocessed chorus
        if len(chorus_whales_data.keys()) > 0:
            print_chorus(chorus_whales_data, f)

        # Remember to write the last whale's sayings in each dialogue to the file
        f.write(what_last_whale_said + ".\n\n")

        # an extra newline before filenames to indicate significant separation
        f.write("\n")
