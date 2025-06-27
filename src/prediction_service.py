import pandas as pd
import joblib

def load_model(model_path):
    """Load a trained model from file."""
    model = joblib.load(model_path)
    return model

def make_prediction(model, input_data):
    """
    Make a fraud prediction.

    Parameters:
    - model: Trained model object
    - input_data: Dict with keys ['TX_AMOUNT', 'HIGH_AMOUNT', 'HOUR', 'DAY']

    Returns:
    - 0 for legitimate, 1 for fraud
    """
   
    df = pd.DataFrame([input_data])

   
    prediction = model.predict(df)

    return int(prediction[0])
