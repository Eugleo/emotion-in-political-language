# Preference-based scoring model for political speech emotionality


This project aims to replicate the analysis of the evolution of emotionality in congressional speeches from [Emotion and reason in political language](https://academic.oup.com/ej/article/132/643/1037/6490125) (G Gennaro and E. Ash, 2021). We use a new methodology based on a transformer scoring model, as opposed to the model based on single-words from the original study.

Our main result is the following timeline plot, which shows the percentage of emotional speeches in different years. The main findings are very similar to the original study, and they even manage to recovers slightly more signal from the speeches, which gives us some evidence as to the utility of the new method. For a full report, please see [this document](assets/report.pdf).

![emotionality timeline](assets/emotionality_timeline_readme.svg)

## Running the code

The project was setup using `pyenv` and `pdm`. If you don't use these tools, make sure you have Python 3.12 and install the dependencies from `pyproject.toml`. E.g. with `venv`,

```
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Depending on which parts of the analysis you want to run, you might need to create a `.env` file with the following variables:

```
# OpenAI API key, used to run GPT to obtain pairwise preferences
OPENAI_API_KEY=
# HuggingFace token, used to download and upload the dataset and the models
HF_TOKEN=
# WandB API key, used to log metrics when training the scoring model
WANDB_API_KEY=
```

Most of the code is contained in the `scripts` folder. If you want to replicate all of the results, you'll need to run the scripts in sequence:

1. `data_preprocess.py`: Parses the speeches and their metadata from the [original source](https://data.stanford.edu/congress_text), and pushes a combined dataset to HuggingFace.
    - You will need to download the `hein-bound.zip` from the [source website](https://data.stanford.edu/congress_text) and move the metadata txt files to `data/original/metadata` and the speech txt files to `data/original/speeches`.
2. `model_pairwise_labels.py`: Samples a given number of speech pairs from the HuggingFace dataset we created in the first step, and submits them to the OpenAI batch API to get preference-based labels about which speech in the pair is more emotional.
    - Run `python scripts/model_pairwise_labels.py submit` to create and submit the batch. This will create a new directory in `data/batch` that stores the batch metadata. It can take up to 24h for OpenAI to process the batch.
    - Run `python scripts/model_pairwise_labels.py download [data/batch/batch_1 data/batch/batch_2 ...]` to attempt to download the given batches. If all the batches have been completed, the script will download the results and push them to HuggingFace.
3. `model_train_classifier.py`: Trains a classifier to predict the emotionality of a speech based on the pairwise labels we obtained in the second step. Pushes the trained model to HuggingFace.
    - Run `python scripts/model_train_classifier.py` to train the classifier. This will create a new directory in `models` that stores the model metadata. It can take up to 24h for the model to train.
4. `model_score.py`: Uses the trained classifier to score a sample of the speeches from the dataset. Saves the results to `data/predictions/predictions.jsonl`.
5. `paper_plots.ipynb`: Generates the plots used in the paper. Downloads the different datasets from HuggingFace and loads `predictions.jsonl`, and saves the plots to `assets`.