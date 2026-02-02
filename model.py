import torch
import torch.nn as nn
from torchvision import models


def build_model(model_name, pretrained=True):
    """
    Builds a PyTorch model for binary classification (Normal vs Anomaly).
    """
    print(f"Building PyTorch model: {model_name}...")

    # 1. Load Base Model
    if model_name == "ResNet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        # Replace the final Fully Connected layer
        model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)  # Output raw logit (we use BCEWithLogitsLoss)
        )

    elif model_name == "DenseNet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        # Replace classifier
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    elif model_name == "EfficientNetB0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        # EfficientNet classifier is a Sequential block; replace the last Linear layer
        model.classifier[1] = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # 2. Freeze Base Layers (Optional: keeps pre-trained features intact)
    # Unfreezing just the custom head for the first training phase
    if pretrained:
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze the new head we just added
        if model_name == "ResNet50":
            for param in model.fc.parameters(): param.requires_grad = True
        elif model_name == "DenseNet121":
            for param in model.classifier.parameters(): param.requires_grad = True
        elif model_name == "EfficientNetB0":
            for param in model.classifier.parameters(): param.requires_grad = True

    return model