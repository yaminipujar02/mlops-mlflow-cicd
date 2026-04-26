import joblib
import pandas as pd

# Load saved model
model = joblib.load("model.pkl")

# Input data
data = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]],
                    columns=[
                        'sepal length (cm)',
                        'sepal width (cm)',
                        'petal length (cm)',
                        'petal width (cm)'
                    ])

# Predict
prediction = model.predict(data)

print("Prediction:", prediction)