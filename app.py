from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import subprocess
import sys

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    global model
    global vectorizer

    message = request.form['message']

    data = vectorizer.transform([message])

    prediction = model.predict(data)[0]

    confidence = max(model.decision_function(data))

    confidence = abs(round(confidence * 10, 2))

    return jsonify({
        'prediction': prediction,
        'confidence': confidence
    })

@app.route('/feedback', methods=['POST'])
def feedback():

    global model
    global vectorizer

    message = request.form['message']
    correct_label = request.form['correct_label']

    feedback_data = {
        'label': [correct_label],
        'text': [message]
    }

    feedback_df = pd.DataFrame(feedback_data)

    try:
        old_feedback = pd.read_csv("feedback.csv")
        feedback_df = pd.concat([old_feedback, feedback_df], ignore_index=True)
    except:
        pass

    feedback_df.to_csv("feedback.csv", index=False)

    # Retrain model automatically using the same Python interpreter
    try:
        subprocess.run([sys.executable, "retrain_model.py"], check=True)
    except Exception:
        # If retraining fails, continue but return an error message to the client
        return jsonify({'message': 'Failed to retrain model'}), 500

    # Reload updated model
    try:
        model = pickle.load(open("model.pkl", "rb"))
        vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    except Exception:
        return jsonify({'message': 'Retrained but failed to reload model'}), 500

    return jsonify({'message': 'Feedback saved and model retrained'})

if __name__ == '__main__':
    app.run(debug=True)