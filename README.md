# 💳 Credit Card Fraud Detection System

An end-to-end Machine Learning project that detects fraudulent credit card transactions using the **Random Forest Classifier**. The project includes data preprocessing, handling class imbalance with **SMOTE**, model training, performance evaluation, and an interactive **Streamlit web application** for real-time fraud prediction.

---

## 🚀 Features

- 📂 Upload transaction CSV file
- 🤖 Fraud detection using Random Forest Classifier
- ⚖️ Balanced the imbalanced dataset using SMOTE (Synthetic Minority Oversampling Technique)
- 📊 Prediction confidence score
- 📈 Interactive dashboard with transaction summary
- 📉 Bar Chart & Pie Chart visualization
- 📥 Download prediction results as CSV
- 🔍 Feature importance analysis
- 🌐 Interactive Streamlit web application

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib
- Streamlit
- Joblib
- Git & GitHub

---

## 📂 Project Structure

```text
Credit_Card_Fraud_Detection/
│── app.py
│── train_model.py
│── fraud_model.pkl
│── columns.pkl
│── encoders.pkl
│── feature_importance.png
│── requirements.txt
│── README.md
│── screenshots/
│    ├── home.png
│    ├── upload.png
│    ├── results.png
│    └── dashboard.png
└── dataset/
     ├── fraudTrain.csv
     └── fraudTest.csv
```

---

## 📊 Machine Learning Workflow

1. Load Dataset
2. Data Preprocessing
3. Label Encoding
4. Train-Test Split
5. Apply SMOTE to balance the training dataset
6. Train Random Forest Classifier
7. Model Evaluation
8. Save Trained Model
9. Deploy with Streamlit

---

## 📈 Model Evaluation

The model is evaluated using:

- ✅ Accuracy
- ✅ Precision
- ✅ Recall
- ✅ F1-Score
- ✅ ROC-AUC Score
- ✅ Confusion Matrix
- ✅ Feature Importance

---

## ▶️ Run Locally

### Clone Repository

```bash
git clone https://github.com/Vanshika-ml/Credit_Card_Fraud_Detection.git
```

### Install Requirements

```bash
pip install -r requirements.txt
```

### Train Model

```bash
python train_model.py
```

### Run Streamlit App

```bash
streamlit run app.py
```

---

## 📸 Project Screenshots

### 🏠 Home Page

![Home](screenshots/home.png)

---

### 📂 Upload Dataset

![Upload](screenshots/upload.png)

---

### 📊 Prediction Results

![Results](screenshots/results.png)

---

### 📈 Dashboard

![Dashboard](screenshots/dashboard.png)

---

## 📌 Dataset

This project uses the **Credit Card Fraud Detection Dataset** from Kaggle.

**Dataset Link:**
https://www.kaggle.com/datasets/kartik2112/fraud-detection

> **Note:** The dataset is not included in this repository because of GitHub file size limitations.

---

## 🌐 Live Demo

🚀 Streamlit App:
_(Add your Streamlit deployment link here after deployment.)_

---

## 🎯 Future Improvements

- XGBoost / LightGBM implementation
- SHAP Explainability
- Hyperparameter Tuning
- Docker Deployment
- REST API Integration
- Real-time Fraud Detection Pipeline

---

## 👩‍💻 Developer

**Vanshika Varshney**

Machine Learning | Data Science | Python Developer

If you found this project useful, don't forget to ⭐ this repository.
