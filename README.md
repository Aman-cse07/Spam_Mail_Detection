# 📧 Spam Mail Detector

A Machine Learning project that detects whether an email or message is **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) techniques and a classification algorithm.

---

## 🌐 Live Server

👉 https://spam-mail-detection-orpin.vercel.app/

---

## 📌 Project Overview

Spam emails are unwanted messages that often contain advertisements, phishing links, scams, or malicious content. This project uses **Machine Learning** to automatically classify incoming messages as **Spam** or **Ham**, helping users filter unwanted emails efficiently.

The model is trained on a labeled dataset of spam and ham messages and predicts the category of new messages entered by the user.

---

## 🎯 Objectives

* Detect spam messages accurately.
* Learn the basics of NLP and text preprocessing.
* Apply Machine Learning algorithms for text classification.
* Build a simple and useful real-world AI application.

---

## 🚀 Features

* 📩 Detects Spam and Ham emails/messages
* 🤖 Machine Learning based classification
* 📝 Text preprocessing using NLP
* ⚡ Fast prediction
* 📊 Easy to understand workflow
* 💻 Beginner-friendly project

---

## 🛠️ Technologies Used

| Technology              | Purpose                  |
| ----------------------- | ------------------------ |
| Python                  | Programming Language     |
| Pandas                  | Data Handling            |
| NumPy                   | Numerical Operations     |
| Scikit-learn            | Machine Learning         |
| NLTK / Regex            | Text Processing          |
| TfidfVectorizer         | Feature Extraction       |
| Multinomial Naive Bayes | Classification Algorithm |
| Jupyter Notebook        | Development Environment  |

---

## 📂 Project Structure

```
Spam-Mail-Detector/
│
├── spam_mail_detector.ipynb
├── mail_data.csv
├── README.md
└── requirements.txt
```

---

## ⚙️ Working Process

### Step 1: Import Libraries

Import all required Python libraries.

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
```

---

### Step 2: Load Dataset

Load the spam email dataset.

```python
data = pd.read_csv("mail_data.csv")
```

---

### Step 3: Data Preprocessing

* Handle missing values
* Convert labels into numerical values
* Clean the text
* Remove unnecessary characters

---

### Step 4: Feature Extraction

Convert text into numerical vectors using **TF-IDF Vectorizer**.

```python
vectorizer = TfidfVectorizer()
```

---

### Step 5: Split Dataset

Divide data into training and testing sets.

```python
train_test_split()
```

---

### Step 6: Train Model

Train the classifier.

```python
MultinomialNB()
```

---

### Step 7: Prediction

Enter a new message and the model predicts:

* 📩 Spam
* ✅ Ham (Not Spam)

---

## 📊 Machine Learning Workflow

```
Dataset
   │
   ▼
Data Cleaning
   │
   ▼
Text Preprocessing
   │
   ▼
TF-IDF Vectorization
   │
   ▼
Train/Test Split
   │
   ▼
Model Training
   │
   ▼
Prediction
```

---

## 📈 Algorithm Used

### Multinomial Naive Bayes

Multinomial Naive Bayes is one of the most popular algorithms for text classification.

### Advantages

* Fast
* Accurate for text data
* Easy to implement
* Works well for spam detection
* Requires less training time

---

## 📚 Natural Language Processing (NLP)

This project uses NLP techniques such as:

* Text Cleaning
* Lowercase Conversion
* Tokenization
* Removing Special Characters
* Feature Extraction using TF-IDF

---

## 📊 Example Prediction

### Input

```
Congratulations!
You have won $1000.
Click the link to claim your prize.
```

### Output

```
Spam
```

---

### Input

```
Hi Aman,
Let's meet tomorrow for the project discussion.
```

### Output

```
Ham (Not Spam)
```

---

## 💡 Applications

* Email Spam Filtering
* SMS Spam Detection
* Social Media Message Filtering
* Customer Support Systems
* Cybersecurity
* Fraud Detection

---

## 📌 Future Improvements

* Deploy using Streamlit
* Create a Flask/Django Web App
* Improve accuracy using Deep Learning
* Add a graphical user interface
* Support multiple languages
* Real-time email detection

---

## ▶️ How to Run

### Clone Repository

```bash
git clone https://github.com/yourusername/Spam-Mail-Detector.git
```

### Navigate

```bash
cd Spam-Mail-Detector
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run

Open the Jupyter Notebook and execute all cells.

---

## 📊 Sample Dataset

The dataset generally contains two columns:

| Label | Message                           |
| ----- | --------------------------------- |
| spam  | Congratulations! You won a prize. |
| ham   | Let's meet tomorrow.              |

---

## 📸 Output

```
Enter Message:

Congratulations!
You won a free lottery.

Prediction:

🚨 Spam Mail
```

---

## 🤝 Contributing

Contributions are always welcome.

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Push to your branch
5. Create a Pull Request

---

## 📖 Learning Outcomes

By completing this project, you will learn:

* Machine Learning Basics
* Natural Language Processing (NLP)
* Text Classification
* TF-IDF Vectorization
* Naive Bayes Algorithm
* Model Training & Testing
* Spam Detection System Development

---

## 📚 References

* Scikit-learn Documentation: https://scikit-learn.org/
* Pandas Documentation: https://pandas.pydata.org/
* NumPy Documentation: https://numpy.org/
* NLTK Documentation: https://www.nltk.org/

---

## 👨‍💻 Author

**Aman Kumar**

**B.Tech CSE (AI & ML)**

**Machine Learning Journey 🚀**

---

## ⭐ Support

If you found this project helpful, don't forget to ⭐ star the repository and share it with others!

Happy Coding! 🚀
