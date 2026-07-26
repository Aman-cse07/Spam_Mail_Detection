import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# Load dataset
print("Loading Dataset...")
df = pd.read_csv("spam.csv")

# Features and labels
X = df["text"]
y = df["label"]

# Convert text into vectors
vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(x_train, y_train)

# Prediction
prediction = model.predict(x_test)
accuracy = accuracy_score(y_test, prediction)

print(f"Accuracy: {accuracy * 100:.2f}%")

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model Saved Successfully")