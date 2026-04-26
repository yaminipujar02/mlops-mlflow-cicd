import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import mlflow
import mlflow.sklearn

# ✅ FIX: set local tracking path (avoids permission error)
mlflow.set_tracking_uri("file:./mlruns")

# Load dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

print("Accuracy:", accuracy)

# MLflow tracking
with mlflow.start_run():
    mlflow.log_param("model", "RandomForest")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")