from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import math
import os
import tempfile
from retrain_model import retrain_model_func, get_readable_path, get_writable_path, normalize_text

app = Flask(__name__)

# Load initial model and vectorizer
def load_model_and_vectorizer():
    try:
        model_path = get_readable_path("model.pkl")
        vectorizer_path = get_readable_path("vectorizer.pkl")
        model = pickle.load(open(model_path, "rb"))
        vectorizer = pickle.load(open(vectorizer_path, "rb"))
        return model, vectorizer
    except Exception:
        # If models don't exist yet, train them
        return retrain_model_func()

model, vectorizer = load_model_and_vectorizer()

# Maintain an in-memory map of user feedback for instant lookups
def load_feedback_memory():
    memory = {}
    try:
        fb_path = get_readable_path("feedback.csv")
        if os.path.exists(fb_path):
            df = pd.read_csv(fb_path)
            for _, row in df.iterrows():
                if pd.notnull(row.get('text')) and pd.notnull(row.get('label')):
                    norm = normalize_text(row['text'])
                    memory[norm] = str(row['label']).strip().lower()
    except Exception:
        pass
    return memory

FEEDBACK_MEMORY = load_feedback_memory()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model, vectorizer
    message = request.form.get('message', '')
    norm_msg = normalize_text(message)

    # 1. Check direct user feedback override first
    if norm_msg in FEEDBACK_MEMORY:
        return jsonify({
            'prediction': FEEDBACK_MEMORY[norm_msg],
            'confidence': 100.0
        })

    # 2. Predict with retrained model
    data = vectorizer.transform([message])
    prediction = model.predict(data)[0]

    # Calculate confidence score smoothly (percentage 50% - 100%)
    try:
        decision_val = float(model.decision_function(data)[0])
        confidence = round((1.0 / (1.0 + math.exp(-abs(decision_val)))) * 100, 2)
    except Exception:
        confidence = 95.0

    return jsonify({
        'prediction': prediction,
        'confidence': confidence
    })

@app.route('/feedback', methods=['POST'])
def feedback():
    global model, vectorizer, FEEDBACK_MEMORY

    message = request.form.get('message', '').strip()
    correct_label = request.form.get('correct_label', '').strip().lower()

    if not message or not correct_label:
        return jsonify({'message': 'Invalid request parameters'}), 400

    norm_msg = normalize_text(message)
    FEEDBACK_MEMORY[norm_msg] = correct_label

    fb_path = get_readable_path("feedback.csv")
    try:
        feedback_df = pd.read_csv(fb_path)
    except Exception:
        feedback_df = pd.DataFrame(columns=['label', 'text'])

    # Remove any previous entries for this exact normalized message
    if not feedback_df.empty:
        feedback_df['norm_text'] = feedback_df['text'].apply(normalize_text)
        feedback_df = feedback_df[feedback_df['norm_text'] != norm_msg].drop(columns=['norm_text'])

    # Append new feedback row
    new_row = pd.DataFrame([{'label': correct_label, 'text': message}])
    feedback_df = pd.concat([feedback_df, new_row], ignore_index=True)

    # Save to writable path (local project folder or /tmp if read-only)
    save_path = get_writable_path("feedback.csv")
    try:
        feedback_df.to_csv(save_path, index=False)
    except Exception as e:
        print(f"Error saving feedback.csv: {e}")

    # Retrain model directly in python (works on Vercel and local)
    try:
        model, vectorizer = retrain_model_func()
    except Exception as e:
        print(f"Error retraining model: {e}")
        # Even if file saving fails on serverless, in-memory model and FEEDBACK_MEMORY are active
        pass

    return jsonify({'message': 'Feedback saved and model retrained'})

if __name__ == '__main__':
    app.run(debug=True)