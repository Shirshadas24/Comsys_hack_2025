# test_script.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from model import SiameseNet
from dataset import FacePairDataset
from sklearn.metrics import accuracy_score, f1_score
import argparse

def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for img1, img2, label in loader:
            img1, img2 = img1.to(device), img2.to(device)
            out = model(img1, img2)
            pred = (out > 0.5).float().cpu().numpy()
            y_pred.extend(pred)
            y_true.extend(label.numpy())
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    return acc, f1

def main(test_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    test_dataset = FacePairDataset(test_path, transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = SiameseNet().to(device)
    model.load_state_dict(torch.load("siamese_model_taskB.pth", map_location=device))

    acc, f1 = evaluate(model, test_loader, device)
    print(f"Test → Accuracy: {acc:.4f}, Macro-F1: {f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test_path", help="Path to validation or test folder")
    args = parser.parse_args()
    main(args.test_path)
