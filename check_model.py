import pickle

# Try loading the model
with open("model/loan_model.pkl", "rb") as f:
    model = pickle.load(f)

# Print the model to confirm
print("✅ Model loaded successfully!")
print(model)
