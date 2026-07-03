import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
# -----------------------
# LOAD DATA
# -----------------------
df = pd.read_csv("dataset/fraudTrain.csv")

y = df["is_fraud"]
X = df.drop("is_fraud", axis=1)

# -----------------------
# SAFE DROP
drop_cols = [
    "Unnamed: 0",
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "first",
    "last",
    "street",
    "city",
    "state",
    "zip",
    "job",
    "dob",
    "trans_num"
]
# -----------------------


X = X.drop(columns=[col for col in drop_cols if col in X.columns])

# -----------------------
# ENCODING + SAVE ENCODERS
# -----------------------
encoders = {}

for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

print("Encoding Done ✔")

# -----------------------
# SPLIT
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------
# APPLY SMOTE
# -----------------------

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE:")
print("X_train shape:", X_train.shape)
print("y_train distribution:")
print(y_train.value_counts())

# -----------------------
# MODEL
# -----------------------
model = RandomForestClassifier(
    n_estimators=80,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Probability
y_prob = model.predict_proba(X_test)[:,1]

# Accuracy
print("\n========== MODEL PERFORMANCE ==========")
print(f"Accuracy:{accuracy_score(y_test,y_pred):.4f}")

#ROC-AUC
print(f"ROC-AUC Score: {roc_auc_score(y_test,y_prob):.4f}")

#Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------
# FEATURE IMPORTANCE
# -----------------------

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# Plot
plt.figure(figsize=(10,6))
plt.barh(
    feature_importance["Feature"][:10],
    feature_importance["Importance"][:10]
)

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Top 10 Important Features")
plt.gca().invert_yaxis()

plt.tight_layout()

# Save graph
plt.savefig("feature_importance.png")

# Show graph
plt.show()

# -----------------------
# SAVE FILES (IMPORTANT)
# -----------------------
joblib.dump(model, "fraud_model.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")
joblib.dump(encoders, "encoders.pkl")   # 🔥 THIS IS IMPORTANT

print("ALL FILES SAVED ✔")
print("fraud_model.pkl + columns.pkl + encoders.pkl")