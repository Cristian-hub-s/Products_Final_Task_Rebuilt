
# Product Category Classification

## Objective
Predict product category from product title using Machine Learning.

## Project Structure

data/
    products.csv

notebooks/
    01_EDA.ipynb
    02_Model_Comparison.ipynb
    03_Final_Model.ipynb

src/
    train_model.py
    predict_category.py

models/
    product_classifier.pkl

## Workflow
1. Exploratory Data Analysis
2. Data Cleaning
3. TF-IDF Feature Extraction
4. Model Comparison
5. Final Model Training
6. Model Serialization

## Training

python src/train_model.py

## Interactive Prediction

python src/predict_category.py

Type a product title and the model predicts its category.

## Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
