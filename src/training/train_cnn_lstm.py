"""
CNN-LSTM with Self-Attention for mental health text classification.

Architecture:
    1. Embedding layer (pre-trained or learned)
    2. 1D CNN for local n-gram feature extraction
    3. Bidirectional LSTM for sequential modeling
    4. Self-Attention mechanism for focusing on key phrases
    5. Fully connected classifier

This is a UNIQUE feature — most student projects don't combine CNN+LSTM+Attention.

Usage:
    python -m src.training.train_cnn_lstm --sample 20000 --epochs 10
"""

import os
import sys
import json
import time
import argparse
import pickle
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.preprocessing.text_cleaner import load_dataset, clean_text


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# VOCABULARY BUILDER
# --------------------------------------------------

class Vocabulary:
    """Simple word-level vocabulary for CNN-LSTM."""

    def __init__(self, max_vocab: int = 50000, min_freq: int = 2):
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}

    def build(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(text.lower().split())

        # Keep top words above min frequency
        common = counter.most_common(self.max_vocab)
        for word, freq in common:
            if freq >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

        print(f"[Vocab] Built vocabulary: {len(self.word2idx):,} words")
        return self

    def encode(self, text, max_length=256):
        tokens = text.lower().split()[:max_length]
        ids = [self.word2idx.get(w, 1) for w in tokens]
        # Pad
        if len(ids) < max_length:
            ids += [0] * (max_length - len(ids))
        return ids


# --------------------------------------------------
# DATASET
# --------------------------------------------------

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length=256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids = self.vocab.encode(self.texts[idx], self.max_length)
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
        }


# --------------------------------------------------
# MODEL: CNN-LSTM WITH SELF-ATTENTION
# --------------------------------------------------

class SelfAttention(nn.Module):
    """Scaled dot-product self-attention over sequence."""

    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key   = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = hidden_size ** 0.5

    def forward(self, x):
        # x: (batch, seq, hidden)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, V)

        return context, attn_weights


class CNN_LSTM_Attention(nn.Module):
    """
    CNN-LSTM with Self-Attention for text classification.

    Architecture:
        Embedding → CNN (multi-kernel) → BiLSTM → Self-Attention → FC → Output
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_classes: int = 6,
        cnn_filters: int = 128,
        kernel_sizes: tuple = (3, 4, 5),
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Multi-kernel CNN
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, cnn_filters, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(cnn_filters),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for k in kernel_sizes
        ])

        # BiLSTM
        cnn_out_dim = cnn_filters * len(kernel_sizes)
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # Self-Attention
        self.attention = SelfAttention(lstm_hidden * 2)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_ids):
        # Embedding: (batch, seq) -> (batch, seq, embed)
        x = self.embedding(input_ids)

        # CNN expects (batch, channels, seq)
        x_t = x.transpose(1, 2)

        # Apply multiple CNN kernels and concatenate
        conv_outs = [conv(x_t) for conv in self.convs]
        x = torch.cat(conv_outs, dim=1)  # (batch, filters*num_kernels, seq)
        x = x.transpose(1, 2)  # (batch, seq, features)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)

        # Self-Attention
        context, attn_weights = self.attention(lstm_out)

        # Global average pooling over sequence
        pooled = context.mean(dim=1)  # (batch, hidden*2)

        # Classify
        logits = self.classifier(pooled)
        return logits


# --------------------------------------------------
# TRAINING LOOP
# --------------------------------------------------

def train_cnn_lstm(
    data_path: str = "data/mental_disorders_reddit.csv",
    output_dir: str = "models/cnn_lstm",
    sample_size: int = None,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    max_length: int = 256,
    patience: int = 3,
):
    print("=" * 60)
    print("  Mental Health System — CNN-LSTM-Attention Training")
    print("=" * 60)
    print(f"  Device:      {DEVICE}")
    print(f"  Epochs:      {epochs}")
    print(f"  Batch Size:  {batch_size}")
    print("=" * 60)

    # Load data
    df, label_encoder = load_dataset(data_path, sample_size=sample_size)
    num_classes = len(label_encoder.classes_)

    texts  = df['text'].tolist()
    labels = df['label_id'].tolist()

    # Build vocabulary
    vocab = Vocabulary(max_vocab=50000, min_freq=2)
    vocab.build(texts)

    # Split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.12, random_state=42, stratify=train_labels
    )

    print(f"[Data] Train: {len(train_texts):,} | Val: {len(val_texts):,} | Test: {len(test_texts):,}")

    # DataLoaders
    train_ds = TextDataset(train_texts, train_labels, vocab, max_length)
    val_ds   = TextDataset(val_texts, val_labels, vocab, max_length)
    test_ds  = TextDataset(test_texts, test_labels, vocab, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Model
    model = CNN_LSTM_Attention(
        vocab_size=len(vocab.word2idx),
        num_classes=num_classes,
    ).to(DEVICE)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[Model] CNN-LSTM-Attention | Parameters: {param_count:,}")

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5)

    best_val_f1 = 0
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(DEVICE)
            labels    = batch['labels'].to(DEVICE)

            logits = model(input_ids)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            if (step + 1) % 100 == 0:
                print(f"  Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | Loss: {total_loss/(step+1):.4f}")

        avg_loss = total_loss / len(train_loader)

        # Validate
        val_loss, val_acc, val_f1 = eval_model(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)

        print(f"\n  Epoch {epoch+1}/{epochs}: Train Loss={avg_loss:.4f} | Val Loss={val_loss:.4f} | "
              f"Val Acc={val_acc:.4f} | Val F1={val_f1:.4f} | Time={time.time()-t0:.0f}s")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            save_cnn_lstm(model, vocab, label_encoder, output_dir)
            print(f"    ✓ Best model saved (F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

    # Final test evaluation
    print("\n" + "=" * 60)
    print("  CNN-LSTM-Attention — Test Set Results")
    print("=" * 60)

    checkpoint = torch.load(os.path.join(output_dir, 'model.pt'), map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            labels    = batch['labels'].to(DEVICE)
            logits = model(input_ids)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    class_names = list(label_encoder.classes_)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    cm = confusion_matrix(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"\n  Test Accuracy: {acc:.4f}")
    print(f"  Test F1 (macro): {f1:.4f}")
    print(f"\n{report}")

    results = {
        'accuracy': acc, 'f1_macro': f1,
        'confusion_matrix': cm.tolist(), 'class_names': class_names,
    }
    with open(os.path.join(output_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results


def eval_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels    = batch['labels'].to(device)
            logits = model(input_ids)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, acc, f1


def save_cnn_lstm(model, vocab, label_encoder, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'vocab': vocab,
        'config': {
            'vocab_size': len(vocab.word2idx),
            'num_classes': len(label_encoder.classes_),
        }
    }, os.path.join(output_dir, 'model.pt'))
    with open(os.path.join(output_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',       default='data/mental_disorders_reddit.csv')
    parser.add_argument('--output',     default='models/cnn_lstm')
    parser.add_argument('--sample',     type=int, default=None)
    parser.add_argument('--epochs',     type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--patience',   type=int, default=3)
    args = parser.parse_args()

    train_cnn_lstm(
        data_path=args.data, output_dir=args.output,
        sample_size=args.sample, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.lr,
        patience=args.patience,
    )
