import os
import re
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# Column constants
OPTION_COLS = ['A', 'B', 'C', 'D', 'E']
OPTION_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
INDEX_TO_OPTION = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

# Candidate paths in order of preference
LOCAL_TRAIN_PATH = "./data/train.csv"
LOCAL_TEST_PATH = "./data/test.csv"
KAGGLE_TRAIN_PATH = "/kaggle/input/competitions/smart-mcq-solver-challenge/train.csv"
KAGGLE_TEST_PATH = "/kaggle/input/competitions/smart-mcq-solver-challenge/test.csv"


def resolve_path(primary_path: str, fallback_path: str, file_type: str) -> str:
    """
    Checks if primary or fallback file paths exist on the current filesystem.
    Works seamlessly across local laptops, Kaggle, and Cloud Deployments.
    """
    if os.path.exists(primary_path):
        return primary_path
    elif os.path.exists(fallback_path):
        return fallback_path
    else:
        raise FileNotFoundError(
            f"\n[ERROR] Could not locate the {file_type} file!\n"
            f"Please ensure '{file_type}.csv' is placed inside the './data/' directory."
        )


def clean_text(text: str) -> str:
    """Removes extra whitespaces and normalizes string inputs."""
    text = str(text)
    return re.sub(r'\s+', ' ', text).strip()


def prepare_data(
    train_path: str = LOCAL_TRAIN_PATH, 
    test_path: str = LOCAL_TEST_PATH, 
    seed: int = 42
):
    """
    Loads, cleans, deduplicates, and stratifies dataset.
    Automatically resolves file paths depending on local or cloud environment.
    """
    actual_train_path = resolve_path(train_path, KAGGLE_TRAIN_PATH, "train")
    actual_test_path = resolve_path(test_path, KAGGLE_TEST_PATH, "test")

    print(f"--> Loading Training Data: {actual_train_path}")
    print(f"--> Loading Test Data:     {actual_test_path}")

    train_df = pd.read_csv(actual_train_path).fillna("")
    test_df = pd.read_csv(actual_test_path).fillna("")

    for col in ['prompt'] + OPTION_COLS:
        train_df[col] = train_df[col].apply(clean_text)
        test_df[col] = test_df[col].apply(clean_text)

    train_df['label'] = train_df['answer'].map(OPTION_TO_INDEX)

    # Deduplicate prompts to avoid data leakage
    train_df = train_df.drop_duplicates(subset=['prompt'] + OPTION_COLS).reset_index(drop=True)

    # Stratified 90/10 train-validation split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    tr_idx, val_idx = next(sss.split(train_df, train_df['label']))

    df_train = train_df.iloc[tr_idx].reset_index(drop=True)
    df_val = train_df.iloc[val_idx].reset_index(drop=True)

    return df_train, df_val, test_df