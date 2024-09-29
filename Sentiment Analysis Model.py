# Step 1: Import libraries
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk
from nltk.corpus import stopwords
from flask import Flask, request, render_template

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Step 2: Load dataset (Assume 'dataset.csv' has 'text' and 'sentiment' columns)
data = pd.read_csv('dataset.csv')

# Step 3: Text Preprocessing
def preprocess_text(text):
    # Remove special characters
    text = re.sub(r'\W', ' ', text)
    
    # Lowercase the text
    text = text.lower()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text

data['cleaned_text'] = data['text'].apply(preprocess_text)

# Step 4: Feature Extraction using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data['cleaned_text']).toarray()
y = data['sentiment']

# Step 5: Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Model Selection (Naive Bayes)
model = MultinomialNB()

# Step 7: Model Training
model.fit(X_train, y_train)

# Step 8: Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

# Step 9: Deployment (Flask Web Interface)
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [preprocess_text(message)]
        vectorized_input_data = tfidf.transform(data).toarray()
        prediction = model.predict(vectorized_input_data)
        return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)






    """ Steps for Deployment:
Install Dependencies: You need to install the required libraries:

pip install pandas numpy scikit-learn flask nltk"""
