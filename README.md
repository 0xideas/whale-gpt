<img src="./assets/Sperm_whale_pod.jpg">

# WhaleGPT: A Transformer Model for Whale Communication

This repository contains **WhaleGPT**, an experimental transformer model designed to explore whale vocalization patterns using data from Sharma et al. Although the data is currently insufficient for practical applications, this project aims to lay the foundation for further research into whale language.

## Overview

WhaleGPT is a decoder-only transformer model that autoregressively predicts sequences of whale vocalizations, also known as **codas**. This project includes the necessary scripts for encoding raw whale click sequences into a structured format suitable for modeling, training the model, and generating predictions.

## Key Concepts

- **Click Sequences**: Whale vocalizations in the Sharma dataset consist of up to 29 clicks, represented by intervals between each click (called inter-click intervals).
- **Codas**: These are subsets of the click sequences, typically consisting of up to 10 clicks (9 intervals). Codas are the core units used to encode and represent the click sequences.
- **Vocalizations**: Longer click sequences are divided into codas, with any surplus clicks treated as ornamentation. The division minimizes the **Manhattan distance** to ensure an accurate representation.

## Methodology

1. **Extracting Codas**:  
   The script `0_extract_codas.py` breaks down whale click sequences into codas, ensuring that the entire sequence is encoded while preserving surplus clicks. The algorithm:
   - Compares the mean relative intervals of codas.
   - Divides the vocalization into codas or treats the excess clicks as ornamentation.
   - Scores these divisions by the Manhattan distance and selects the encoding with the lowest total distance.

2. **Creating Whale Dialogues**:  
   The script `1_create_dialogue.py` generates dialogues from overlapping coda sequences. Key decisions include:
   - **Simultaneity**: Codas starting within 0.3 seconds of each other are considered simultaneous and placed in a single row.
   - **Multiple Whales**: When multiple whales are present, a "me" vs. "other" encoding is applied. Each whale's codas are represented separately in the columns:
     - `Coda1`, `Ornamentation1`, `Duration1` for the primary whale.
     - `Coda2`, `Ornamentation2`, `Duration2` for other whales.  
     Silence is represented by the token `98`.

3. **Training the Model**:  
   WhaleGPT is a transformer model with 126k parameters. It uses the previous 25 values of all six variables (`Coda1`, `Ornamentation1`, `Duration1`, `Coda2`, `Ornamentation2`, `Duration2`) to predict the next increment in these sequences.

4. **Data**:  
   The dataset used for training is located in `data/whale-dialogues.csv`, consisting of around 9,000 observations. The generated whale dialogues can be found in `outputs/predictions/sequifier-dialogue-best-5000-predictions.csv`.

## Current Limitations

Due to the limited data size, the model tends to generate repetitive outputs, where both whales emit the coda "5" either while the other is silent or simultaneously. Despite this, the model serves as a starting point for further experimentation and highlights the need for more extensive whale vocalization data.

## Setup & Reproducibility

To reproduce the development of WhaleGPT, follow these steps. (Note: these steps should work on Mac or most Linux distributions.)

### 1. Environment Setup

```bash
conda create --name whale-gpt python=3.11 -y
conda activate whale-gpt
pip install sequifier==0.4.0.0 scikit-learn
```

### 2. Run Preprocessing Scripts

```bash
# Compute mean relative intervals for codas
python scripts/00_create_coda_means.py

# Extract codas from raw click sequences
python scripts/0_extract_codas.py

# Create dialogue data from overlapping coda sequences
python scripts/1_create_dialogue.py
```

### 3. Model Training and Inference

```bash
# Preprocess the data
sequifier preprocess

# Train the transformer model
sequifier train

# Generate predictions
sequifier infer
```

## Future Directions

This project demonstrates the feasibility of applying transformer models to whale vocalizations, but further research is required. Specifically:
- **Data Collection**: More whale vocalization data is needed to improve the model's accuracy.
- **Model Refinement**: Experiment with model architecture and hyperparameters to better capture the intricacies of whale communication.
