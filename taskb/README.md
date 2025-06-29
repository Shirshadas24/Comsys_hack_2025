# Face Matching with Siamese Network

##  Overview
This project uses a Siamese CNN to perform face verification on distorted and reference face image pairs.
Validation → Accuracy: 0.9504, Macro-F1: 0.8206

##  Folder Structure
Comys_Hackathon5/
├── Task_B/
│ ├── train/
│ └── val/



##  Model Architecture
- CNN-based embedding extractor
- Distance-based matching (absolute diff)
- Binary classification (match or not)

##  How to Run

1. **Train** using the Colab notebook: `siamese_net_taskB.ipynb`
2. **Test**:
```bash
python test_script.py /path/to/val
```
🧾 Output
The script prints:

Accuracy

Macro F1-Score

## Files
model.py: Model architecture

dataset.py: Custom dataset

test_script.py: Evaluation

siamese_model_taskB.pth:  Download Pretrained Weights  
[Siamese Model Weights (Google Drive)] (https://drive.google.com/file/d/1qp_jY6h0cDai10RfJ_HgbvTBZXIO1J3h/view?usp=drivesdk)

README.md: You are here

siamese_model_diagram.png: Model architecture diagram

