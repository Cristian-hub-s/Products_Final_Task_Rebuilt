import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/products.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Remove missing values
df = df.dropna(
    subset=[
        "Product Title",
        "Category Label"
    ]
)

print("Dataset shape after cleaning:")
print(df.shape)

# Features and target
X = df["Product Title"].astype(str)
y = df["Category Label"]

# Remove rare categories (optional)
category_counts = y.value_counts()

rare_classes = category_counts[
    category_counts < 2
].index

df = df[
    ~df["Category Label"].isin(rare_classes)
]

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

print("\nTrain size:", X_train.shape)
print("Test size:", X_test.shape)