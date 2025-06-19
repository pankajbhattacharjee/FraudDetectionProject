from flask import Flask, request, jsonify
from src.prediction_service import load_model, make_prediction

# Create Flask app
app = Flask(__name__)

# Load model when server starts
model = load_model('model/fraud_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Check for required keys
    required_keys = ['TX_AMOUNT', 'HIGH_AMOUNT', 'HOUR', 'DAY']
    if not all(key in data for key in required_keys):
        return jsonify({'error': 'Missing input fields'}), 400

    # Make prediction
    result = make_prediction(model, data)

    return jsonify({
        'prediction': 'Fraud' if result == 1 else 'Legit'
    })

if __name__ == '__main__':
    app.run(debug=True)
