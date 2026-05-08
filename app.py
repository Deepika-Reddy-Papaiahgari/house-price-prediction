from flask import Flask, render_template, request
import pickle
import numpy as np

# Create Flask app
app = Flask(__name__)

# Load trained model
model = pickle.load(open('models/house_model.pkl', 'rb'))

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    # Get form values
    features = [float(x) for x in request.form.values()]

    # Convert into numpy array
    final_features = np.array([features])

    # Predict
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template(
        'index.html',
        prediction_text=f'Predicted House Price: ${output}'
    )

# Run app
if __name__ == "__main__":
    app.run(debug=True)