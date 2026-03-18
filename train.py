"""
train.py - Model Training Pipeline

Description:
    This module handles the complete training pipeline for anomaly detection models.
    It manages data loading, model training, validation, and model persistence.
    Supports multi-GPU training, data augmentation, and real-time progress tracking.

Purpose:
    - Load and preprocess training and validation datasets
    - Train models with configurable hyperparameters
    - Track training metrics and validation accuracy
    - Save best performing models to cache directory
    - Support multiple architectures and datasets
    - Provide progress callbacks for UI integration

Author: ImageMetrics Project Team
Created: 2026-03-18
Version: 1.0.0-alpha
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import build_model
from metrics_manager import MetricsManager

# --- CONFIGURATION ---
CACHE_DIR = 'cache'  # <--- NEW FOLDER
IMG_SIZE = 150
DATASET_DIR = 'K:\ImageDataset'

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model_pipeline(model_name, epochs, batch_size, progress_callback=None, status_text=None, dataset_choice='ucirvine_chest_xray'):
    device = get_device()
    gpu_count = torch.cuda.device_count()
    metrics = MetricsManager()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if status_text:
        if gpu_count > 1:
            status_text.text(f"🚀 Training {model_name} on {gpu_count} GPUs!")
        else:
            status_text.text(f"Training {model_name} on {device}...")

    # --- Data Transforms ---
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # --- Load Data ---
    # Use dataset_choice parameter to determine dataset folder
    data_dir = os.path.join(DATASET_DIR, dataset_choice)
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # --- Build Model ---
    model = build_model(model_name).to(device)

    if gpu_count > 1:
        model = nn.DataParallel(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- Training Loop ---
    metrics.start_timer()
    best_acc = 0.0
    total_steps = epochs * len(train_loader)
    current_step = 0

    # Define save path in cache folder
    save_path = os.path.join(CACHE_DIR, f"{model_name}_best.pth")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            current_step += 1

            if progress_callback:
                progress_callback.progress(current_step / total_steps)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(images)
                preds = torch.sigmoid(outputs) > 0.5
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        avg_loss = running_loss / len(train_loader)

        msg = f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2%}"
        print(msg)
        if status_text: status_text.text(msg)

        if val_acc > best_acc:
            best_acc = val_acc
            # Save to CACHE_DIR
            if gpu_count > 1:
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)

    duration = metrics.stop_timer()
    metrics.save_training_stats(model_name, duration, best_acc)

    return save_path, best_acc, duration


if __name__ == "__main__":
    train_model_pipeline("DenseNet121", 5, 32)