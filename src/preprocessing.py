import re
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

OPTION_COLS = ['A', 'B', 'C', 'D', 'E']
OPTION_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
INDEX_TO_OPTION = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}

def clean_text(text: str) -> str:
    """Removes extra whitespaces from text."""
    text = str(text)
    return re.sub(r'\s+', ' ', text).strip()

def prepare_data(train_path: str = "./data/train.csv", test_path: str = "./data/test.csv", seed: int = 42):
    """Loads, cleans, deduplicates, and splits the data into train and validation sets."""
    train_df = pd.read_csv(train_path).fillna("")
    test_df = pd.read_csv(test_path).fillna("")

    for col in ['prompt'] + OPTION_COLS:
        train_df[col] = train_df[col].apply(clean_text)
        test_df[col] = test_df[col].apply(clean_text)

    train_df['label'] = train_df['answer'].map(OPTION_TO_INDEX)

    # Deduplicate prompts to avoid data leakage
    train_df = train_df.drop_duplicates(subset=['prompt'] + OPTION_COLS).reset_index(drop=True)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    tr_idx, val_idx = next(sss.split(train_df, train_df['label']))

    df_train = train_df.iloc[tr_idx].reset_index(drop=True)
    df_val = train_df.iloc[val_idx].reset_index(drop=True)

    return df_train, df_val, test_df