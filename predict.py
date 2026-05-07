import pickle
import numpy as np

# Load model
with open("models/house_model.pkl", "rb") as f:
    model = pickle.load(f)

# Example input
sample_data = np.array([[8.3252, 41, 6.984127, 1.02381,
                         322, 2.555556, 37.88, -122.23]])

prediction = model.predict(sample_data)

print("Predicted House Price:", prediction[0])