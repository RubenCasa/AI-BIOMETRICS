# AI Biomechanical Pose Correction & Error Pattern Detection
**Hybrid 1D-CNN + Bidirectional LSTM Neural Network | HTML5 Standalone Web Dashboard**

![Training Curves](training_curves.png)

---

## 1. Project Overview & Composition
This repository contains an end-to-end deep learning and computer vision system designed to automatically evaluate human exercise technique from video sequences, identify postural errors (spine/back desalignment vs. knee/limb collapse), and generate natural language diagnostic feedback.

### 📁 Repository Structure
```text
AI_Pose_Correction_CNN_LSTM/
│
├── dashboard.html             # Standalone HTML5/JS Interactive Dashboard (Visualizes skeletons & MP4 upload)
├── real_data.js               # Penn Action real sequence frames, 13-joint coordinates & AI predictions
├── abrir_dashboard.bat        # Windows quick-launcher executable for dashboard.html
├── main.py                    # Unified CLI entry point for training and evaluation
├── requirements.txt           # Python package dependencies (PyTorch, NumPy, Matplotlib, Scikit-learn)
├── training_curves.png        # Publication-grade model training and validation curves
│
└── src/                       # Modular PyTorch Deep Learning Package
    ├── __init__.py
    ├── config.py              # Hyperparameters (Features=26, Classes=3, Window=46 frames)
    ├── dataset.py             # Penn Action DataLoader, Z-score temporal normalization & augmentation
    ├── model.py               # PoseQualityHybridModel (1D-CNN Spatial Extractor + BiLSTM Temporal Layer)
    ├── train.py               # Training loop with Adam optimizer, L2 Weight Decay & Early Stopping
    ├── feedback.py            # Natural Language Generation (NLG) engine for explainable AI feedback
    └── visualize_demo.py      # Video generation utility for presentation demos
```

---

## 2. Dataset Attribution & Source
The real human exercise kinematics and visual sequences processed in this project are derived from the official **Penn Action Dataset**, a benchmark video dataset annotated with 13 anatomical keypoints per frame across real-world physical actions.

* **Official Dataset Repository & Reference:**  
  [Penn Action Dataset Official Page](https://dreamdragon.github.io/PennAction/?spm=a2ty_o01.29997173.0.0.5ed555fbfff7Af)

* **Keypoint Annotations Used:**  
  Each sequence tracks $(x, y)$ frame-by-frame coordinates for **13 anatomical joints**: Head, Shoulders, Elbows, Wrists, Hips, Knees, and Ankles across 15 sports and gym exercise categories (Squat, Pushup, Bench Press, Situp, Clean & Jerk, etc.).

---

## 3. Deep Learning Architecture (`src/model.py`)
Our neural network processes continuous time-series input vectors $\mathbf{X} \in \mathbb{R}^{B \times 26 \times 46}$ ($26$ normalized joint coordinates across $46$ frames):

1. **Spatial Convolutional Block (`1D-CNN`)**:
   - Two `nn.Conv1d` layers extract local spatial relationships, implicit angular velocities, and inter-joint coordination features.
   - Regularized via `BatchNorm1d` and `Dropout(0.3)`.
2. **Temporal Recurrent Block (`BiLSTM`)**:
   - A Bidirectional LSTM layer models long-range sequential dependencies across the full exercise execution cycle (eccentric descent $\to$ concentric ascent).
3. **Probabilistic Classifier & NLG**:
   - Projects hidden states to 3 posture quality classes (**Class 0:** Correct Posture, **Class 1:** Back/Spine Error, **Class 2:** Limb/Knee Error) and automatically emits natural language feedback explaining technical corrections.

---

## 4. Getting Started

### Launching the Interactive Web Dashboard
No server or Python backend is required to inspect the real sequences and evaluate custom videos:
- Double-click **`abrir_dashboard.bat`** or open **`dashboard.html`** directly in any web browser.
- **Custom MP4 Evaluation:** Click the **`📹 EVALUAR MI PROPIO VIDEO (MP4)`** button inside the sidebar to upload and inspect any personal exercise video in real-time.

### Training the PyTorch Model
To train the hybrid model from scratch using the optimized hyperparameters (`lr=5e-4`, `weight_decay=5e-4`, `dropout=0.3`):

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run model training with Early Stopping
python main.py --mode train --epochs 35 --batch_size 32
```
