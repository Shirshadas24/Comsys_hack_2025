# Face Matching with Siamese Network

##  Overview
This project uses a Siamese CNN to perform face verification on distorted and reference face image pairs.
Validation → Acc=96.10% | Prec=97.55% | Rec=98.24% | F1=97.89%

##  Folder Structure
```

taskb/
├── task_b_siamese_net.ipynb ← Training notebook
├── test_script.py ← Test script to evaluate performance
├── model.py ← Siamese model architecture
├── dataset.py ← Dataset class to generate training pairs
├── siamese_model_taskB.pth ← Trained model weights (external download recommended)
├── siamese_model_diagram.png ← Network diagram
├── requirements.txt ← Dependency file
└── README.md ← This file

```


##  Model Architecture
- CNN-based embedding extractor
- Distance-based matching (absolute diff)
- Binary classification (match or not)

##  How to Run

### 1. Install dependencies
```
pip install -r requirements.txt
```
2. **Train** using the Colab notebook: `siamese_net_taskB.ipynb`
3. **Test**:
```
python test_script.py /path/to/val
```
🧾 Output
The script prints:



Accuracy  
Precision  
Recall  
F1-Score 

## Files
model.py: Model architecture

dataset.py: Custom dataset

test_script.py: Evaluation

siamese_model_taskB.pth:  Download Pretrained Weights  
[Siamese Model Weights (Google Drive)] (https://drive.google.com/file/d/1kG7_-5Ylov9_YTrBJ51_G75j66yEL16U/view?usp=sharing)

README.md: You are here

siamese_model_diagram.png: Model architecture diagram

requirements.txt : Dependency file

