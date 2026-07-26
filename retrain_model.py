import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

# Original dataset
main_df = pd.read_csv("spam.csv")

# Feedback dataset
try:
    feedback_df = pd.read_csv("feedback.csv")

    combined_df = pd.concat([main_df, feedback_df], ignore_index=True)

except:
    combined_df = main_df

# Features and labels
X = combined_df["text"]
y = combined_df["label"]

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# Train model (use SGDClassifier to replace deprecated PassiveAggressiveClassifier)
model = SGDClassifier(loss='hinge', penalty=None, learning_rate='pa1', eta0=1.0, max_iter=1000)
model.fit(X_vectorized, y)

# Save updated model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model Retrained Successfully")