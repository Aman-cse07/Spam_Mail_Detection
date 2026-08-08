import pandas as pd
import pickle
import os
import tempfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

def get_readable_path(filename):
    """Returns readable path for filename, preferring /tmp if present."""
    tmp_path = os.path.join(tempfile.gettempdir(), filename)
    if os.path.exists(tmp_path):
        return tmp_path
    local_path = os.path.join(os.getcwd(), filename)
    if os.path.exists(local_path):
        return local_path
    return filename

def get_writable_path(filename):
    """Returns writable path for filename (local dir if writable, else /tmp)."""
    local_path = os.path.join(os.getcwd(), filename)
    try:
        test_file = os.path.join(os.getcwd(), ".write_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        return local_path
    except (OSError, IOError, PermissionError):
        return os.path.join(tempfile.gettempdir(), filename)

def normalize_text(text):
    return ' '.join(str(text).split())

def retrain_model_func():
    # Load dataset
    main_csv_path = get_readable_path("spam.csv")
    main_df = pd.read_csv(main_csv_path)

    # Load feedback dataset
    feedback_csv_path = get_readable_path("feedback.csv")
    try:
        feedback_df = pd.read_csv(feedback_csv_path)
    except Exception:
        feedback_df = pd.DataFrame(columns=["label", "text"])

    if not feedback_df.empty:
        # Normalize text for accurate deduplication
        feedback_df['norm_text'] = feedback_df['text'].apply(normalize_text)
        # Keep the latest label for each unique normalized text
        feedback_df = feedback_df.drop_duplicates(subset=['norm_text'], keep='last')
        
        # Filter out matching texts from main_df to resolve conflicts
        main_df['norm_text'] = main_df['text'].apply(normalize_text)
        main_df = main_df[~main_df['norm_text'].isin(feedback_df['norm_text'])]
        
        # Clean temporary columns
        feedback_clean = feedback_df[['label', 'text']]
        main_clean = main_df[['label', 'text']]

        # Weight feedback 5x to ensure strong influence of user corrections
        feedback_weighted = pd.concat([feedback_clean] * 5, ignore_index=True)
        combined_df = pd.concat([main_clean, feedback_weighted], ignore_index=True)
    else:
        combined_df = main_df[['label', 'text']]

    # Features and labels
    X = combined_df["text"]
    y = combined_df["label"]

    # Vectorization
    vectorizer = TfidfVectorizer(stop_words='english')
    X_vectorized = vectorizer.fit_transform(X)

    # Train SGDClassifier
    model = SGDClassifier(loss='hinge', max_iter=1000, random_state=42)
    model.fit(X_vectorized, y)

    # Save updated model and vectorizer
    model_path = get_writable_path("model.pkl")
    vectorizer_path = get_writable_path("vectorizer.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Model Retrained Successfully: saved to {model_path} and {vectorizer_path}")
    return model, vectorizer

if __name__ == '__main__':
    retrain_model_func()