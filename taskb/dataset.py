# dataset.py
import os
import random
from torch.utils.data import Dataset
from PIL import Image
import torch

class FacePairDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.identity_folders = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])
        self.samples = []

        for person in self.identity_folders:
            ref_path = os.path.join(root_dir, person)
            ref_img = [i for i in os.listdir(ref_path) if i.endswith('.jpg')][0]
            ref_img_path = os.path.join(ref_path, ref_img)
            distorted_folder = os.path.join(ref_path, 'distortion')
            distorted_imgs = [os.path.join(distorted_folder, f) for f in os.listdir(distorted_folder)]

            for img in distorted_imgs:
                self.samples.append((ref_img_path, img, 1))

            neg_candidates = [p for p in self.identity_folders if p != person]
            neg_person = random.choice(neg_candidates)
            neg_path = os.path.join(root_dir, neg_person)
            neg_img = [i for i in os.listdir(neg_path) if i.endswith('.jpg')][0]
            self.samples.append((ref_img_path, os.path.join(neg_path, neg_img), 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.samples[idx]
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)