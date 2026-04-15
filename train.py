import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# Load data
df = pd.read_csv("AAPL.csv")

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# 🔍 Debug
print("Last 5 Close Values:")
print(df['Close'].tail())

data = df[['Close']].values

# Scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create sequences
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(scaled_data, time_step)

# Split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Model
model = Sequential([
    Dense(128, activation='relu', input_dim=time_step),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train
history = model.fit(
    X_train, y_train,
    epochs=60,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# Inverse transform
train_pred = scaler.inverse_transform(train_pred)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1,1))

test_pred = scaler.inverse_transform(test_pred)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))

# R2 Score
train_r2 = r2_score(y_train_actual, train_pred)
test_r2 = r2_score(y_test_actual, test_pred)

print(f"✅ Train R2 Score: {train_r2}")
print(f"✅ Test R2 Score: {test_r2}")

# Plot Loss
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Loss Graph")
plt.show()

# Plot Actual vs Predicted
plt.figure()
plt.plot(y_test_actual, label='Actual')
plt.plot(test_pred, label='Predicted')
plt.legend()
plt.title("Actual vs Predicted")
plt.show()

# Save
model.save("model.h5")
joblib.dump(scaler, "scaler.pkl")

print("🔥 Model trained & saved successfully!")