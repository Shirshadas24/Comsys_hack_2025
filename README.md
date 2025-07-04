#  COMSYS Hackathon 2025 Submission

**Authors:** [Shirsha Das](https://shirshadas.vercel.app/)  , [Pritam Kumar Roy](https://pritam-kumar-roy.vercel.app/)

**Track:** Computer Vision  

**Tasks:** Gender Classification (Task A) & Face Matching (Task B)

---

##  Repository Structure

```
comsys-hackathon/
├── taska/
│ ├── taskA_gender_classifier.ipynb ← Training & validation notebook
│ ├── test_script.py ← Test script
│ ├── efficientnet_gender_classifier.pth ← Trained model weights 
│ ├── efficientnet_gender_diagram.png ← Model diagram
| ├── requirements.txt ← Dependencies
│ └── README.md ← Task-specific instructions
| 
| 
│
├── taskb/
│ ├── task_b_siamese_net.ipynb ← Training & validation notebook
│ ├── test_script.py ← Test script
│ ├── model.py ← Siamese model definition
│ ├── dataset.py ← Dataset class for training pairs
│ ├── siamese_model_taskB.pth ← Trained Siamese weights (external link provided)
│ ├── siamese_model_diagram.png ← Model diagram
| ├── requirements.txt ← Dependencies
│ └── README.md ← Task-specific instructions

```
---

##  Task A: Gender Classification

- **Model Used:** EfficientNet-B0 (fine-tuned)
- **Type:** Binary Classification
- **Goal:** Predict gender (`male`, `female`) from face images

###  Dataset

- `train/` and `val/` folders  
  ├── `male/`  
  └── `female/`

###  Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-Score  

###  Files Explained

| File | Description |
|------|-------------|
| `taskA_gender_classifier.ipynb` | Colab notebook to train and test |
| `test_script.py` | Standalone test runner for a test dataset |
| `efficientnet_gender_classifier.pth` | Trained weights (can be shared via GDrive if >100MB) |
| `efficientnet_gender_diagram.png` | Architecture diagram |

---

##  Task B: Face Matching with Distorted Inputs

- **Model Used:** Siamese Network with CNN-based encoders
- **Type:** Face Verification (not classification)
- **Goal:** Match distorted face images to correct person folders

###  Dataset

- `train/` folder with identity-wise folders
- `val/` folder with test images (match vs non-match)

###  Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-Score  

###  Files Explained

| File | Description |
|------|-------------|
| `task_b_siamese_net.ipynb` | Training & validation notebook |
| `test_script.py` | Evaluation script for test set |
| `model.py` | Siamese network definition |
| `dataset.py` | Dataset class generating face pairs |
| `siamese_model_taskB.pth` | Trained model weights |
| `siamese_model_diagram.png` | Siamese model architecture |

---

##  Installation

To run locally:

```
pip install torch torchvision scikit-learn efficientnet_pytorch matplotlib
```
Optional (for model visualization):
```
pip install torchviz graphviz
```
▶ Run Test Script
To evaluate on your own test set (Task A or B):

```
python test_script.py /path/to/test/folder
```
Output will include:
```
Accuracy: ...
Precision: ...
Recall: ...
F1-Score: ...
```

## Model Weights Download (if needed)
Due to GitHub's 100MB limit, siamese_model_taskB.pth file is available here:

[Google Drive Link](https://drive.google.com/file/d/1kG7_-5Ylov9_YTrBJ51_G75j66yEL16U/view?usp=sharing)


## Submission Notes:

All notebooks and scripts have been tested on Google Colab (T4 GPU).

.pth files are either uploaded via Git or hosted externally.

Code is modular and reproducible.

Diagrams created using torchviz.

© 2025 Shirsha Das – COMSYS Hackathon
