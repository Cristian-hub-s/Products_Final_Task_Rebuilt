
import pickle

with open("models/product_classifier.pkl","rb") as f:
    model = pickle.load(f)

while True:
    title = input("Product title: ")
    if title.lower() == "exit":
        break

    print("Predicted category:", model.predict([title])[0])
