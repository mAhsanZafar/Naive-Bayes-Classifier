import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Load the dataset
file_path = "genders.csv"  # Ensure the file is in the same directory
df = pd.read_csv(file_path)

# Extract features (last two letters of names)
df["LastTwo"] = df["name"].str[-2:].str.lower()

# Encode labels
X = df["LastTwo"].values
y = df["gender"].values

# Convert text features into numerical features
vectorizer = CountVectorizer(analyzer='char', ngram_range=(2,2))
classifier = MultinomialNB()

# Create a pipeline (vectorization + classification)
model = make_pipeline(vectorizer, classifier)

# Train the model
model.fit(X, y)

# Function to predict gender
def predict_gender(name):
    last_two = name[-2:].lower()
    prediction = model.predict([last_two])[0]
    return prediction
while True:
    # Get user input
    test_name = input("Enter a name (or type 'exit' to quit): ").strip()
    
    # Exit condition
    if 'exit' in test_name.lower():  # Allow case-insensitive 'exit'
        print("Exiting program. Goodbye!")
        break
    
    # Predict gender
    predicted_gender = predict_gender(test_name)
    
    # Output result
    print(f"Predicted Gender for '{test_name}': {predicted_gender}")
