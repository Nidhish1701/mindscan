"""
Text preprocessing and data loading for the Mental Health Early Warning System.

Handles:
    - Text cleaning (HTML, URLs, special chars, anonymization)
    - Dataset loading from CSV
    - Train/val/test splitting
    - PyTorch DataLoader creation for transformer training
"""

import re
import os
import html
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------

# Patterns for anonymization
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
PHONE_PATTERN = re.compile(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')
URL_PATTERN   = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
USERNAME_PATTERN = re.compile(r'u/[A-Za-z0-9_-]+|@[A-Za-z0-9_-]+')


def clean_text(text: str) -> str:
    """
    Clean and normalize text for model input.

    Steps:
        1. Decode HTML entities
        2. Remove URLs, emails, usernames (anonymization)
        3. Remove excess whitespace
        4. Lowercase
        5. Remove very short texts
    """
    if not isinstance(text, str) or len(text.strip()) < 3:
        return ""

    # Decode HTML entities
    text = html.unescape(text)

    # Remove URLs
    text = URL_PATTERN.sub('[URL]', text)

    # Anonymize emails and phone numbers
    text = EMAIL_PATTERN.sub('[EMAIL]', text)
    text = PHONE_PATTERN.sub('[PHONE]', text)

    # Remove Reddit usernames
    text = USERNAME_PATTERN.sub('[USER]', text)

    # Remove [removed] and [deleted] markers
    text = re.sub(r'\[(removed|deleted)\]', '', text, flags=re.IGNORECASE)

    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Skip very short texts
    if len(text) < 10:
        return ""

    return text


# --------------------------------------------------
# DATASET LOADING
# --------------------------------------------------

# Label mapping for the 6 subreddit classes
LABEL_MAP = {
    'Anxiety':       'Anxiety',
    'BPD':           'BPD',
    'bipolar':       'Bipolar',
    'depression':    'Depression',
    'mentalillness': 'Mental Illness',
    'schizophrenia': 'Schizophrenia',
}

# Reverse map for display
LABEL_DESCRIPTIONS = {
    'Anxiety':        'Generalized Anxiety Disorder indicators',
    'BPD':            'Borderline Personality Disorder indicators',
    'Bipolar':        'Bipolar Disorder indicators',
    'Depression':     'Major Depressive Disorder indicators',
    'Mental Illness': 'General mental health concerns',
    'Schizophrenia':  'Schizophrenia spectrum indicators',
}


def load_dataset(
    csv_path: str = "data/mental_disorders_reddit.csv",
    text_col: str = "selftext",
    label_col: str = "subreddit",
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Load and preprocess the mental health Reddit dataset.

    Args:
        csv_path:     Path to the CSV file.
        text_col:     Column containing the post text.
        label_col:    Column containing the subreddit label.
        sample_size:  If set, sample this many rows (for quick testing/Colab).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (cleaned DataFrame with 'text' and 'label' columns, fitted LabelEncoder)
    """
    print(f"[Data] Loading dataset from {csv_path} ...")
    df = pd.read_csv(csv_path, usecols=[text_col, label_col], engine='c')
    print(f"[Data] Raw dataset: {len(df):,} rows")

    # Combine title + selftext if both exist
    if 'title' in df.columns and text_col == 'selftext':
        df['text'] = df['title'].fillna('') + ' ' + df[text_col].fillna('')
    else:
        df['text'] = df[text_col].fillna('')

    # Map labels
    df['label'] = df[label_col].map(LABEL_MAP)
    df = df.dropna(subset=['label'])

    # Sample if requested (DO THIS BEFORE CLEANING to save time)
    if sample_size and sample_size < len(df):
        print(f"[Data] Sampling {sample_size:,} rows BEFORE cleaning...")
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    # Clean text
    print("[Data] Cleaning text ...")
    df['text'] = df['text'].apply(clean_text)
    df = df[df['text'].str.len() > 0].reset_index(drop=True)

    # Encode labels
    le = LabelEncoder()
    df['label_id'] = le.fit_transform(df['label'])

    print(f"[Data] Final dataset: {len(df):,} rows, {len(le.classes_)} classes")
    print(f"[Data] Classes: {list(le.classes_)}")
    print(f"[Data] Distribution:\n{df['label'].value_counts().to_string()}")

    return df[['text', 'label', 'label_id']], le


# --------------------------------------------------
# PYTORCH DATASET
# --------------------------------------------------

class MentalHealthDataset(Dataset):
    """PyTorch Dataset for mental health text classification."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
        )
        return {
            'input_ids':      encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long),
        }


def prepare_dataloaders(
    df: pd.DataFrame,
    tokenizer,
    batch_size: int = 32,
    max_length: int = 128,
    test_size: float = 0.15,
    val_size: float = 0.10,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split data and create PyTorch DataLoaders for train/val/test.

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    texts  = df['text'].tolist()
    labels = df['label_id'].tolist()

    # Split: train+val vs test
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    # Split: train vs val
    relative_val = val_size / (1 - test_size)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels, test_size=relative_val,
        random_state=random_state, stratify=train_val_labels
    )

    print(f"[Data] Train: {len(train_texts):,} | Val: {len(val_texts):,} | Test: {len(test_texts):,}")

    train_ds = MentalHealthDataset(train_texts, train_labels, tokenizer, max_length)
    val_ds   = MentalHealthDataset(val_texts,   val_labels,   tokenizer, max_length)
    test_ds  = MentalHealthDataset(test_texts,  test_labels,  tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader
