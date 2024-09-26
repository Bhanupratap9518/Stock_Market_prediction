# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Set plot styles
sns.set(style='whitegrid')
%matplotlib inline

# Load the dataset
df = pd.read_csv("NSE-TATA.csv")
df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
df.set_index("Date", inplace=True)

# Plot the closing price history
plt.figure(figsize=(16, 8))
plt.plot(df["Close"], label='Close Price History')
plt.title('Tata Stock Price History')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Data preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['Close']].values)

# Create training and validation datasets
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
valid_data = scaled_data[int(len(scaled_data) * 0.8):]

# Prepare the training data
x_train, y_train = [], []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
x_train = np.array(x_train)
y_train = np.array(y_train)

# Reshape data for LSTM [samples, time steps, features]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=50, return_sequences=False))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units=1))  # Output layer

# Compile the model
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Set callbacks for early stopping and model checkpointing
early_stopping = EarlyStopping(monitor='loss', patience=5)
model_checkpoint = ModelCheckpoint("best_lstm_model.h5", save_best_only=True)

# Train the model
lstm_model.fit(x_train, y_train, epochs=50, batch_size=32, callbacks=[early_stopping, model_checkpoint])

# Prepare the validation data
inputs_data = scaled_data[len(scaled_data) - len(valid_data) - 60:]
inputs_data = inputs_data.reshape(-1, 1)
inputs_data = scaler.transform(inputs_data)

# Create test data
X_test = []
for i in range(60, inputs_data.shape[0]):
    X_test.append(inputs_data[i-60:i, 0])
X_test = np.array(X_test)

# Reshape test data for LSTM
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Make predictions
predictions = lstm_model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Visualize the results
train_data = df[:int(len(df) * 0.8)]
valid_data = df[int(len(df) * 0.8):]
valid_data['Predictions'] = predictions

plt.figure(figsize=(16, 8))
plt.plot(train_data['Close'], label='Training Data')
plt.plot(valid_data[['Close', 'Predictions']], label='Validation Data and Predictions')
plt.title('Tata Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
