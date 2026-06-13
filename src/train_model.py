import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

print("Loading dataset...")

df = pd.read_csv("data/products.csv")

# Curățarea numelor coloanelor
df.columns = df.columns.str.strip()

print("\nColumns found:")
print(df.columns.tolist())

# Verificare coloane necesare
required_columns = ["Product Title", "Category Label"]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(
            f"Missing required column: {col}\n"
            f"Available columns: {df.columns.tolist()}"
        )

# Normalizare etichete
df["Category Label"] = df["Category Label"].replace({
    "CPU": "CPUs",
    "Mobile Phone": "Mobile Phones",
    "fridge": "Fridges"
})

# Eliminare valori lipsă
df = df.dropna(subset=["Product Title", "Category Label"])

print(f"\nDataset shape: {df.shape}")
print(f"Number of categories: {df['Category Label'].nunique()}")

# Features și target
X = df["Product Title"].astype(str)
y = df["Category Label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining model...")

# Pipeline
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
    ),
    (
        "classifier",
        LinearSVC()
    )
])

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, predictions)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")
print(f"Accuracy: {accuracy:.4f}")
print()

print(classification_report(
    y_test,
    predictions
))

# Creare folder models dacă nu există
os.makedirs("models", exist_ok=True)

# Salvare model
model_path = "models/product_classifier.pkl"

with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"\nModel saved successfully:")
print(model_path)


