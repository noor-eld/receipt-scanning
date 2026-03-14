# train_model.py
# Trains a TF-IDF + Logistic Regression classifier to distinguish food from non-food.
#
# WHAT IS TF-IDF?
# TF-IDF (Term Frequency - Inverse Document Frequency) converts text into numbers.
# It looks at how important each word is. "Chicken" appears often in food items but
# rarely in non-food — so it gets a high score for food. "Soap" gets a high score
# for non-food. The model learns these patterns from the training data.
#
# WHAT IS LOGISTIC REGRESSION?
# It's a simple ML model that draws a line between two categories.
# Given the TF-IDF numbers, it learns: "items on THIS side of the line are food,
# items on THAT side are non-food." Simple, fast, and effective for text classification.
#
# HOW TO RUN:
#   python train_model.py
#   This creates a file called "food_classifier.pkl" that the main app loads.

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import pickle

# Step 1: Load training data
print("Loading training data...")
df = pd.read_csv("data/training_data.csv")
print(f"  Total samples: {len(df)}")
print(f"  Food: {len(df[df['label'] == 'food'])}")
print(f"  Non-food: {len(df[df['label'] == 'non_food'])}")

# Step 2: Prepare features (X) and labels (y)
# X = the product names (text)
# y = the labels (food or non_food)
X = df["product_name"]
y = df["label"]

# Step 3: Create the ML pipeline
# A pipeline chains steps together: TF-IDF → Logistic Regression
# This way, when we predict later, both steps run automatically.
pipeline = Pipeline([
    # TF-IDF Vectorizer: converts text to numbers
    # - ngram_range=(1,2) means it looks at single words AND pairs of words
    #   e.g., "chicken breast" is captured as both "chicken", "breast", AND "chicken breast"
    # - lowercase=True normalizes everything
    # - analyzer="word" splits on words (not characters)
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        lowercase=True,
        analyzer="word",
    )),

    # Logistic Regression: the actual classifier
    # - max_iter=1000 gives it enough time to learn
    # - C=1.0 controls how strict the model is (default works well here)
    ("classifier", LogisticRegression(
        max_iter=1000,
        C=1.0,
    )),
])

# Step 4: Evaluate with cross-validation
# Cross-validation splits the data into 5 parts, trains on 4, tests on 1, rotates.
# This tells us how accurate the model really is (not just on training data).
print("\nEvaluating model with 5-fold cross-validation...")
scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
print(f"  Accuracy per fold: {[f'{s:.2%}' for s in scores]}")
print(f"  Average accuracy: {scores.mean():.2%}")

# Step 5: Train on ALL data (now that we know it works)
print("\nTraining final model on all data...")
pipeline.fit(X, y)

# Step 6: Save the trained model to a file
# pickle serializes the Python object to a file so we can load it later
# without retraining every time the server starts.
model_path = "food_classifier.pkl"
with open(model_path, "wb") as f:
    pickle.dump(pipeline, f)
print(f"Model saved to {model_path}")

# Step 7: Quick test with example items
print("\n--- Quick Test ---")
test_items = [
    "Chicken Breast Fillet",
    "GV PNT BUTTR",
    "Toilet Cleaner",
    "BNLS BRST MD",
    "Banana Loose",
    "Nitrile Gloves",
    "Sriracha Sce",
    "Shampoo Anti Dandruff",
    "Fusilli HWG",
    "12 CT NITRIL",
]

for item in test_items:
    prediction = pipeline.predict([item])[0]
    probability = pipeline.predict_proba([item]).max()
    print(f"  {item:30s} → {prediction:10s} ({probability:.1%} confident)")