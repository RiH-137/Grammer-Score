# 🎯 Grammar Score Prediction from Audio

This project builds a machine learning pipeline to predict grammar proficiency scores (1 to 5) from `.wav` audio recordings of spoken English. It is based on the rubric for evaluating grammar in speech, and uses supervised learning techniques to train and evaluate the model.

---

## 📁 Dataset Description

### Audio Folders:
- `audios_train/`: Audio files for training (used with `train.csv`)
- `audios_test/`: Audio files for testing/inference (used with `test.csv`)

### CSV Files:
- `train.csv`: Contains filenames and labels (grammar scores 1–5) for training
- `test.csv`: Contains filenames only (used for inference/validation)
- `sample_submission.csv`: Template for submission (`filename`, `label`)

---

## 🧠 Project Goal

To build a robust machine learning model that can analyze spoken audio and predict a speaker's grammar proficiency score, as per the rubric:

| Score | Description |
|-------|-------------|
| 1 | Struggles with basic grammar and sentence construction |
| 2 | Makes consistent basic grammar errors |
| 3 | Adequate grammar, but with frequent structural issues |
| 4 | Strong grammar with minor issues |
| 5 | Excellent grammatical control and self-correction |

---

## 🔧 Project Pipeline

### 1. **EDA (Exploratory Data Analysis)**
- Visualize distribution of grammar scores in training set
- Inspect audio durations and waveform statistics

### 2. **Audio Preprocessing**
- Load `.wav` files with `librosa`
- Normalize audio to fixed sample rate and duration
- Extract MFCC (Mel Frequency Cepstral Coefficients) as features

### 3. **Feature Engineering**
- Convert MFCC arrays to fixed-length vectors (mean + std per MFCC)
- Optionally add delta and delta-delta MFCCs

### 4. **Model Development**
- Train regression models (Ridge, Random Forest, etc.)
- Evaluate with MAE, RMSE
- Convert predictions to integers (1–5) for classification metrics

### 5. **Model Evaluation**
- Use rounded predictions for:
  - **Accuracy**
  - **Confusion Matrix**
  - **Classification Report**
- Use test dataset (with known labels hidden) as validation set

### 6. **Hyperparameter Tuning**
- Use `GridSearchCV` for Ridge/Lasso
- Try different MFCC configurations (e.g., `n_mfcc=13`, `40`)

### 7. **Prediction and Inference**
- Predict grammar scores for `audios_test/`
- Format predictions into `submission.csv`

### 8. **Reusable Prediction Function**
- Given any `.wav` file, output its grammar score prediction using the trained model

---

## 🧪 Evaluation Metrics

- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **Accuracy** (after rounding predictions)
- **Confusion Matrix**
- **Classification Report** (Precision, Recall, F1-score)

---

## 📦 Requirements

- Python 3.7+
- `librosa`
- `scikit-learn`
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `joblib` (for model saving)

---

## Accuracy

- MAE: 0.43
- RMSE: 0.61
- Accuracy: 78.6%
