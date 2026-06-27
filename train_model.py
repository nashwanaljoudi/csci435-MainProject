# Train a Random Forest classifier on the processed ASL landmark dataset
# and save the model to model/classifier.pkl for use in cv/06_classify.py.

import pickle
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the processed landmark dataset
with open("data/dataset.pkl", "rb") as f:
    data = pickle.load(f)

X, y = np.array(data["X"]), np.array(data["y"])

# Split into 80% train / 20% test, stratified so every class is represented
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train the Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# Evaluate and print results
y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print(classification_report(y_test, y_pred))

# Save the trained model for the classifier to load at runtime
Path("model").mkdir(exist_ok=True)
joblib.dump(clf, "model/classifier.pkl")
print("Model saved to model/classifier.pkl")
