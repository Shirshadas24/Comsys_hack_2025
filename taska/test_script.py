# test_script.py
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse

def evaluate(model, dataloader, device):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device).float()
            outputs = model(inputs)
            predictions = torch.sigmoid(outputs).cpu().numpy() > 0.5
            preds.extend(predictions.astype(int).flatten())
            labels.extend(targets.cpu().numpy().astype(int))

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")

def main(test_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(root=test_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = EfficientNet.from_name('efficientnet-b0')
    model._fc = nn.Linear(model._fc.in_features, 1)
    model.load_state_dict(torch.load('efficientnet_gender_classifier.pth', map_location=device))
    model = model.to(device)

    evaluate(model, test_loader, device)

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python test_script.py <test_data_path>")
    else:
        main(sys.argv[1])
