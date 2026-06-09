from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the model safely
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model (2).pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    # We will embed the HTML directly into a template or simple string for layout layout
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the form submission
        beds = float(request.form.get('beds'))
        baths = float(request.form.get('baths'))
        lot_size = float(request.form.get('lot_size'))
        zip_code = float(request.form.get('zip_code'))
        
        # Structure the data matching the model's feature order: [beds, baths, lot_size, zip_code]
        features = np.array([[beds, baths, lot_size, zip_code]])
        
        # Generate prediction
        prediction = model.predict(features)[0]
        
        # Format output as local currency layout (assuming USD, adjust if needed)
        formatted_prediction = f"${prediction:,.2f}"
        
        return render_template('index.html', 
                               prediction_text=f'Estimated Value: {formatted_prediction}',
                               beds=beds, baths=baths, lot_size=lot_size, zip_code=int(zip_code))
    
    except Exception as e:
        return render_template('index.html', error_text=f"Error processing prediction: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
