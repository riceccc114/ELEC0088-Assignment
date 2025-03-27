import csv
import requests
import datetime
import statistics
import numpy as np
import joblib
import pandas as pd
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import time
from datetime import datetime, timedelta
# load weather data

data = pd.read_csv('new_data.csv')
data=data.dropna()
#Automatically generate lagged target columns
data['humidity_day1'] = data['humidity'].shift(-1)
data['humidity_day2'] = data['humidity'].shift(-2)
data['humidity_day3'] = data['humidity'].shift(-3)
data=data.dropna()

X = data[['date','mean_temp','precipitation','pressure','windspeed']] #features
y = data[['humidity_day1','humidity_day2','humidity_day3']]#Target

#Normalization
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
#Scale to the range of 0 to 1
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

#Split training set and test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)


#Building a fully connected neural network
model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(3))
model.compile(optimizer='adam', loss='mse')
model.summary()

#train the model
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2)
#model.evaluate(X_test, y_test)


joblib.dump(scaler_X,'scaler_X3.pkl')
joblib.dump(scaler_y,'scaler_y3.pkl')
model.save('humidity_predictor_model.h5')

y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test = scaler_y.inverse_transform(y_test)

for i in range(3):
    mse_i = mean_squared_error(y_test[:, i], y_pred[:, i])
    print(f"Day {i+1} RMSE: {mse_i ** 0.5:.2f}")
#########################################################################################################################
