# 🧬 Blood Cell Cancer Detection – Hackathon Project

## 🚀 Project Overview
This project aims to detect **blood cancer** using microscopic images of blood cells. Leveraging **deep learning** (CNNs), **transfer learning** (EfficientNet), and **hyperparameter optimization** (Optuna), this model classifies cell types to assist in early diagnosis.

We built an end-to-end pipeline for training, evaluation, logging (MLflow), and deployment (Streamlit).

---

## 📂 Dataset
- **Source**: [Kaggle Dataset - Blood Cell Images](https://www.kaggle.com/datasets/sumithsingh/blood-cell-images-for-cancer-detection)
- **Classes**:
  - `basophil`
  - `erythroblast`
  - `monocyte`
  - `myeloblast`
  - `seg_neutrophil`

- **Structure**: Images stored in class-named folders
- **Total images**: ~5,000

---

## 🧼 Data Preprocessing
- Dataset split: **Train (80%)**, **Validation (10%)**, **Test (10%)**
- Class imbalance handled via stratified splits
- Image resizing to `(128, 128)`
- Normalization: Rescaled to [0, 1]
- Color histogram feature extraction + PCA + KMeans (for EDA)

---

## 🏗️ Model Architecture
- Transfer Learning via `EfficientNetB3` (pretrained on ImageNet)
- Custom classification head:
  - `BatchNormalization`
  - `Dense(256, ReLU)` with regularization
  - `Dropout(0.45)`
  - `Dense(num_classes, softmax)`

---

## 🧪 Training
- **Optimizer**: Adamax
- **Loss**: Categorical Crossentropy
- **Epochs**: 10
- **Augmentation**: rotation, zoom, flip

---

## 🎯 Performance
- **Train Accuracy**: ~95.5%
- **Validation Accuracy**: ~93.2%
- **Test Accuracy**: ~94.2%

Classification Report shows strong precision/recall for each class.

---

## 📊 Hyperparameter Optimization (Optuna)
- **Cross-validation**: StratifiedKFold (k=3)
- **Optimized Parameters**:
  - Number of filters in Conv2D layer
  - Dense layer units
  - Dropout rate

- **Tracking**: Integrated with MLflow
- **Best accuracy**: 93.4%
- **Visualization**: Param importances, optimization history

---

## 🧠 Model Logging & Export
- Framework: `MLflow`
- Logged Artifacts:
  - Trained Keras model (.h5)
  - Converted TFLite model
  - Hyperparameters & metrics
  - Confusion Matrix

---

## 🌐 Web App (Streamlit)
- Upload blood cell image
- Predict cell type with confidence scores
- Show bar chart of class probabilities
- Save results as `.csv`
- View prediction history
- Deployed via **Ngrok** tunnel

🔗 **Try it out**: `https://<your-ngrok-url>.ngrok-free.app`

---

## 🧪 Inference Example
```bash
✅ Prédiction : erythroblast (40.46% de confiance)
```

---

## 📦 Dependencies
- Python 3.11+
- TensorFlow 2.18+
- OpenCV
- Optuna
- MLflow
- Streamlit
- pyngrok
- scikit-learn

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 👩‍💻 Team
- Data Engineer: Natali Ilyan - Ferdaous Bechraoui - Neila Bouali
- Data Scientis: Natali Ilyan - Ferdaous Bechraoui - Neila Bouali
- Deep Learning Engineer: Natali Ilyan - Ferdaous Bechraoui - Neila Bouali
- Visualization Expert: Natali Ilyan - Ferdaous Bechraoui - Neila Bouali
---

## 📁 Project Structure
```
├── app.py                     # Streamlit app
├── Bloods.h5                 # Saved Keras model
├── Bloods.tflite             # TFLite version
├── best_optuna_params.csv   # Exported best parameters
├── confusion_matrix.png     # Visualized confusion matrix
├── id2label.json            # Class index mapping
├── mlruns/                  # MLflow tracking directory
└── README.md
```

---

## 📌 Notes
- All code tested and executed in Google Colab
- MLflow accessible via ngrok
- Project reproducible via notebook and app

---


