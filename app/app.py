from flask import Flask, request, render_template
import joblib
import pandas as pd
import webbrowser
import threading

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Convert inputs to float
        values = [float(x) for x in request.form.values()]

        # Check if 4 values entered
        if len(values) != 4:
            return render_template("index.html", prediction="❌ Please enter exactly 4 values")

        data = pd.DataFrame([values])
        prediction = model.predict(data)[0]

        # Convert prediction to label
        if prediction == 0:
            result = "Setosa 🌸"
        elif prediction == 1:
            result = "Versicolor 🌼"
        else:
            result = "Virginica 🌺"

        return render_template("index.html", prediction=result)

    except:
        # Handle invalid input
        return render_template("index.html", prediction="❌ Invalid input! Enter numeric values only")


if __name__ == "__main__":
    # Auto open browser
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(debug=True)