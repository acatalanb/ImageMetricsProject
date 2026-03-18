"""
metrics_manager.py - Performance Metrics and Evaluation

Description:
    This module provides comprehensive metrics calculation and visualization for
    binary classification models. It handles timing, performance metrics computation,
    confusion matrix generation, and training statistics persistence.

Purpose:
    - Calculate classification metrics (accuracy, precision, recall, F1-score)
    - Generate and visualize confusion matrices
    - Track training and evaluation timing
    - Save and load training statistics
    - Provide formatted time display
    - Generate classification reports

Metrics Supported:
    - Accuracy: Overall correctness of predictions
    - Precision: Proportion of positive predictions that are correct
    - Recall: Proportion of actual positives correctly identified
    - F1-Score: Harmonic mean of precision and recall
    - Confusion Matrix: True/False Positive/Negative breakdown

Author: ImageMetrics Project Team
Created: 2026-03-18
Version: 1.0.0-alpha
"""

import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)


class MetricsManager:
    def __init__(self):
        self.start_time = 0
        self.end_time = 0

    def start_timer(self):
        self.start_time = time.time()

    def stop_timer(self):
        self.end_time = time.time()
        return self.end_time - self.start_time

    def format_time(self, seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h)}h {int(m)}m {int(s)}s" if h else f"{int(m)}m {int(s):.2f}s"

    def calculate_metrics(self, y_true, y_pred_probs):
        """
        Calculates metrics.
        y_true: list or array of actual labels (0 or 1)
        y_pred_probs: list or array of probabilities (0.0 to 1.0)
        """
        # Convert probabilities to binary predictions (Threshold 0.5)
        y_pred = [1 if p > 0.5 else 0 for p in y_pred_probs]

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),  # tolist for JSON serializing
            "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        }

    def save_training_stats(self, model_name, duration, val_accuracy, filepath='cache/training_stats.json'):
        stats = {
            "model_name": model_name,
            "training_duration": self.format_time(duration),
            "final_accuracy": float(val_accuracy),
            "timestamp": time.ctime()
        }
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=4)

    def load_training_stats(self, filepath='cache/training_stats.json'):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    def plot_confusion_matrix(self, cm, labels=['Normal', 'Anomaly']):
        fig = plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        return fig