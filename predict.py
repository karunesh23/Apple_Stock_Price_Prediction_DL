import numpy as np
import joblib
from tensorflow.keras.models import load_model

model = load_model("model.h5")
scaler = joblib.load("scaler.pkl")

def predict_price(last_60_days):
    data = np.array(last_60_days).reshape(-1, 1)

    scaled_data = scaler.transform(data)

    X_input = scaled_data.reshape(1, -1)

    prediction = model.predict(X_input, verbose=0)

    predicted_price = scaler.inverse_transform(prediction)

    return float(predicted_price[0][0])