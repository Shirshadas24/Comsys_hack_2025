# Task A: Gender Classification with EfficientNet

## Overview
This project uses `EfficientNet-B0` to classify facial images into Male or Female categories using a binary classifier.

Test Results → Accuracy: 0.9550, Precision: 0.9656, Recall: 0.9748, F1-Score: 0.9702

##  Model
- Pretrained EfficientNet-B0 (from `efficientnet_pytorch`)
- Final layer: Fully connected → sigmoid output for binary classification
- Loss: `BCEWithLogitsLoss`
- Optimizer: Adam

##  Dataset
```
Comys_Hackathon5/
├── Task_A/
│ ├── train/
│ │ ├── male/
│ │ └── female/
│ └── val/
|  ├── male/
|  └── female/
```

##  How to Run
### 1. Install dependencies
```
pip install -r requirements.txt
```
2. Train the model
Run the notebook: gender_classifier.ipynb

3. Evaluate
```
python test_script.py /path/to/val
```
## Output
The script will print:

Accuracy

Precision

Recall

F1-Score

## Files
gender_classifier.ipynb: Model training and validation notebook

efficientnet_gender_classifier.pth: Pretrained weights

test_script.py: Evaluation

README.md: You are here 

efficientnet_gender_diagram: Model architecture diagram

requirements.txt : Dependencies
