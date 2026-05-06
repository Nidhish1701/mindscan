"""
DistilBERT / RoBERTa fine-tuning for mental health text classification.

Features:
    - Mixed precision training (fp16) for GPU acceleration
    - Early stopping with patience
    - Model checkpointing (saves best model)
    - Learning rate scheduling with warmup
    - Gradient accumulation for effective larger batch sizes
    - Generates confusion matrix, classification report
    - Exports accuracy metrics and plots

Usage:
    python -m src.training.train_transformer --sample 20000 --epochs 4 --batch_size 32
"""

import os
import sys
import json
import time
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.preprocessing.text_cleaner import load_dataset, prepare_dataloaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_transformer(
    data_path: str = "data/mental_disorders_reddit.csv",
    model_name: str = "distilbert-base-uncased",
    output_dir: str = "models/distilbert/best_model",
    sample_size: int = None,
    epochs: int = 4,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    patience: int = 2,
    fp16: bool = True,
    grad_accum_steps: int = 1,
):
    """
    Fine-tune a DistilBERT/RoBERTa model for mental health classification.

    Returns:
        dict with accuracy, f1, classification report, and confusion matrix
    """
    print("=" * 60)
    print("  Mental Health Early Warning System — Transformer Training")
    print("=" * 60)
    print(f"  Model:       {model_name}")
    print(f"  Device:      {DEVICE}")
    print(f"  Epochs:      {epochs}")
    print(f"  Batch Size:  {batch_size}")
    print(f"  LR:          {learning_rate}")
    print(f"  FP16:        {fp16 and DEVICE.type == 'cuda'}")
    print(f"  Output:      {output_dir}")
    print("=" * 60)

    # ---- Load Data ----
    df, label_encoder = load_dataset(data_path, sample_size=sample_size)
    num_labels = len(label_encoder.classes_)

    # ---- Load Tokenizer + Model ----
    print(f"\n[Model] Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    ).to(DEVICE)
    print(f"[Model] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- DataLoaders ----
    train_loader, val_loader, test_loader = prepare_dataloaders(
        df, tokenizer, batch_size=batch_size, max_length=max_length
    )

    # ---- Optimizer + Scheduler ----
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs // grad_accum_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    # ---- Mixed Precision ----
    use_fp16 = fp16 and DEVICE.type == 'cuda'
    scaler = GradScaler(enabled=use_fp16)

    # ---- Training Loop ----
    best_val_f1 = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_f1': []}

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['labels'].to(DEVICE)

            with autocast(enabled=use_fp16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accum_steps

            if (step + 1) % 100 == 0:
                elapsed = time.time() - t0
                print(f"  Epoch {epoch+1}/{epochs} | Step {step+1}/{len(train_loader)} | "
                      f"Loss: {total_loss/(step+1):.4f} | {elapsed:.0f}s")

        avg_train_loss = total_loss / len(train_loader)
        train_time = time.time() - t0

        # --- Validate ---
        val_loss, val_acc, val_f1_score, _, _ = evaluate(model, val_loader, DEVICE)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['val_f1'].append(val_f1_score)

        print(f"\n  Epoch {epoch+1}/{epochs} Summary:")
        print(f"    Train Loss: {avg_train_loss:.4f} | Time: {train_time:.0f}s")
        print(f"    Val Loss:   {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1_score:.4f}")

        # --- Early Stopping ---
        if val_f1_score > best_val_f1:
            best_val_f1 = val_f1_score
            patience_counter = 0
            # Save best model
            save_model(model, tokenizer, label_encoder, output_dir, history)
            print(f"    ✓ New best model saved (F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"    ✗ No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

    # ---- Final Evaluation on Test Set ----
    print("\n" + "=" * 60)
    print("  Final Evaluation on Test Set")
    print("=" * 60)

    # Reload best model
    model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(DEVICE)

    test_loss, test_acc, test_f1, all_preds, all_labels = evaluate(model, test_loader, DEVICE)

    # Classification report
    class_names = list(label_encoder.classes_)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n  Test Accuracy: {test_acc:.4f}")
    print(f"  Test F1 (macro): {test_f1:.4f}")
    print(f"\n  Classification Report:\n{report}")
    print(f"\n  Confusion Matrix:")
    print(f"  Classes: {class_names}")
    for i, row in enumerate(cm):
        print(f"    {class_names[i]:<20} {row}")

    # Save results
    results = {
        'accuracy': test_acc,
        'f1_macro': test_f1,
        'classification_report': report_dict,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
        'history': history,
        'config': {
            'model_name': model_name,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'max_length': max_length,
            'sample_size': sample_size,
            'device': str(DEVICE),
        }
    }

    results_path = os.path.join(output_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    # Save confusion matrix plot
    try:
        save_confusion_matrix_plot(cm, class_names, output_dir)
    except Exception as e:
        print(f"  [Warning] Could not save confusion matrix plot: {e}")

    return results


def evaluate(model, dataloader, device):
    """Evaluate model on a dataloader. Returns loss, accuracy, f1, predictions, labels."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, acc, f1, np.array(all_preds), np.array(all_labels)


def save_model(model, tokenizer, label_encoder, output_dir, history=None):
    """Save model, tokenizer, and label encoder to disk."""
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    le_path = os.path.join(output_dir, 'label_encoder.pkl')
    with open(le_path, 'wb') as f:
        pickle.dump(label_encoder, f)

    # Save label map as JSON for easy inspection
    label_map = {str(i): str(c) for i, c in enumerate(label_encoder.classes_)}
    with open(os.path.join(output_dir, 'label_map.json'), 'w') as f:
        json.dump(label_map, f, indent=2)

    if history:
        with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)


def save_confusion_matrix_plot(cm, class_names, output_dir):
    """Save confusion matrix as a PNG image."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title('Confusion Matrix — Mental Health Classification', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix plot saved to {plot_path}")


# --------------------------------------------------
# CLI
# --------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DistilBERT for mental health classification")
    parser.add_argument('--data',       default='data/mental_disorders_reddit.csv', help='Path to CSV dataset')
    parser.add_argument('--model_name', default='distilbert-base-uncased',          help='HuggingFace model name')
    parser.add_argument('--output',     default='models/distilbert/best_model',     help='Output directory')
    parser.add_argument('--sample',     type=int, default=None,                     help='Sample N rows (for quick testing)')
    parser.add_argument('--epochs',     type=int, default=4,                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,                       help='Batch size')
    parser.add_argument('--lr',         type=float, default=2e-5,                   help='Learning rate')
    parser.add_argument('--max_length', type=int, default=128,                      help='Max token length')
    parser.add_argument('--patience',   type=int, default=2,                        help='Early stopping patience')
    parser.add_argument('--no_fp16',    action='store_true',                        help='Disable mixed precision')
    args = parser.parse_args()

    train_transformer(
        data_path=args.data,
        model_name=args.model_name,
        output_dir=args.output,
        sample_size=args.sample,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        patience=args.patience,
        fp16=not args.no_fp16,
    )
