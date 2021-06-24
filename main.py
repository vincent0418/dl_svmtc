# -*- coding: utf-8 -*-
"""DL Project shareprice.ipynb

Original file is located at
    https://colab.research.google.com/drive/19Kk4ggeWOveo9e5DDOrrsrWb1v6bgFei

"""

import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential, load_model


# Preprocessing
def load_data(company, start, end):
    data = pd.read_csv('data/CAC40_stocks_2010_2021.csv')
    data = data[np.in1d(data[['CompanyName']], company)]
    data = data.loc[(data['Date'] >= start) & (data['Date'] < end)]
    return data


# Company to be focused on
company = 'Groupe PSA'

data = load_data(company=company,
                 start='2012, 1, 1',
                 end='2019, 1, 1')

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# Build LSTM model
def LSTM_model():
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))

    return model


# Training
model = LSTM_model()
model.summary()
model.compile(optimizer='adam',
              loss='mean_squared_error')

# Define callbacks

# Save weights only for best model
checkpointer = ModelCheckpoint(filepath='weights_best.hdf5',
                               verbose=2,
                               save_best_only=True)

model.fit(x_train,
          y_train,
          epochs=50,
          batch_size=32,
          callbacks=[checkpointer])

# Save model in file
model.save("model.h5")

# Load mode from file
loaded_model = load_model("model.h5")

# Predictions
# test model accuracy on existing data
test_data = load_data(company=company,
                      start='2019-1-1',
                      end=dt.datetime.now().strftime("%y-%m-%d"))

actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

x_test = []
for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x - prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = loaded_model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

plt.plot(actual_prices, color='black', label=f"Actual {company} price")
plt.plot(predicted_prices, color='green', label=f"predicted {company} price")
plt.title(f"{company} share price")
plt.xlabel("time")
plt.ylabel(f"{company} share price")
plt.legend()
plt.show()

# predicting next day
real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs + 1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = loaded_model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"prediction: {prediction}")
