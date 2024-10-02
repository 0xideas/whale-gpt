Hello!

This repo is a first attempt to build WhaleGPT, a transformer 'language' model for whale language. The data and codas used in this model come from this work from [Sharma et al.](https://github.com/pratyushasharma/sw-combinatoriality). At this point the data is likely insufficient for a directly useful outcome, so the model has to be considered experimental.

The approach is the following:

1. The data used in Sharma et al. contains click sequences with a length of up to 28 inter click intervals, i.e. 29 clicks, but the codas being used to encode these click sequences have up to 9 inter click intervals, i.e. up to 10 clicks. Longer click sequences can not be encoded as a single coda, as throwign away surplus click intervals would be rash. We have therefore developed an algorithm to code click sequences into codas, contained in [`scripts/0_extract_codas.py`](https://github.com/0xideas/whale-gpt/blob/main/scripts/0_extract_codas.py). This algorithm is used to decode all click sequences, now named 'vocalizations', into sequences of codas. This applies to all vocalizations, including those consisting of 10 clicks or less. The algorithm takes the mean relative interval sequence for each coda (i.e. the mean time percentile of each click in the coda), and recursively divides the vocalization into codas or 'surplus' clicks (that can usually be interpreted as ornamentation). The sequences of these subdivisions are then scored by the manhattan distance, and the sequence of codas with the minimum total distance is then taken as the vocalization encoding into codas.
2. These coda sequences are then encoded into dialogue form using [`scripts/1_create_dialogue.py`](https://github.com/0xideas/whale-gpt/blob/main/scripts/1_create_dialogue.py). The main decisions on how to represent these overlapping coda sequences by multiple whales in discrete form are (1) Codas that begin sufficiently close in time are considered simultaneous and encoded in a single row. The threshold used to making this determination is somewhat arbitrary and currently set to 0.3 seconds. (2) Some recordings contain two or more whales, and this can be represented in multiple ways. Here, we adopt a 'me' vs 'other' encoding, where for each whale contained in a recording, the codas emitted by the whale is encoded in the columns "Coda1", "Ornamentation1" and "Duration1", while the codas emitted by any of the other whales are encoded in the columns "Coda2", "Ornamentation2" and "Duration2". When either the primary whale or the other whales are silent, this is encoded with an additional token '98'.
3. The language model itself is a decoder only transformer with 126k parameters that autoregressively models "Coda1", "Ornamentation1", "Duration1", "Coda2", "Ornamentation2" and "Duration2". Each incremental output of these variables is generated from the previous 25 values of all of these variables. We use the package [sequifier](https://github.com/0xideas/sequifier) that enables the easy configuration, training and inference for models of this type.

Several generated "Whale dialogues" can be found in [`outputs/predictions/sequifier-dialogue-best-5000-predictions.csv`](https://github.com/0xideas/whale-gpt/blob/main/outputs/predictions/sequifier-dialogue-best-5000-predictions.csv), where each sequenceId value is a single "dialogue". Currently most of them end up with both whales repeatedly emitting the coda "5", either while the other is silent or together. 

The roughly 9k observations contained in this dataset are clearly insufficient to create a useful model, but this modelling work should serve as an encouragement for additional data collection and a basis for future model development.

The development of this model can be reproduced in the following steps, using Mac (or likely most Linux distributions). Since all the artefacts are also contained in this repository, all the steps after 3. can also be executed individually.

1. `conda create --name whale-gpt python=3.11 -y`
2. `conda activate whale-gpt`
3. `pip install sequifier==0.4.0.0 scikit-learn`
4. `python scripts/00_create_coda_means.py`
5. `python scripts/0_extract_codas.py`
6. `python scripts/1_create_dialogue.py`
7. `sequifier preprocess`
8. `sequifier train`
9. `sequifier infer`