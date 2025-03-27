#Simple udp socket server
import socket
import sys
import csv
import string
import csv
import chardet
import requests
import datetime
import statistics
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import time
from datetime import datetime, timedelta
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model


HOST = ''
PORT = 8888
# Datagram (udp) socket

max_temp_model = load_model('max_temp_predictor_model.h5')
min_temp_model = load_model('min_temp_predictor_model.h5')
humidity_model = load_model('humidity_predictor_model.h5')
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')
scaler_X2 = joblib.load('scaler_X2.pkl')
scaler_y2 = joblib.load('scaler_y2.pkl')
scaler_X3 = joblib.load('scaler_X3.pkl')
scaler_y3 = joblib.load('scaler_y3.pkl')


def get_weather(city, api_key,fun):
    url = "https://api.weatherapi.com/v1/current.json"
    params = {
        'key': api_key,
        'q': city,
        'lang': 'en'
    }
    response = requests.get(url, params=params)
    data = response.json()
    if fun==0:#temp
        try:
            date = datetime.strptime(data['current']['last_updated'], "%Y-%m-%d %H:%M")
            date_num=int(int(date.strftime('%Y%m%d%H%M'))%100000000)/10000
            #print(data['current'])
            return [int(date_num),data['current']['humidity'],data['current']['precip_mm'],data['current']['pressure_mb'],data['current']['wind_kph']]
        except KeyError:
            raise Exception("Weather data extraction failed")
    elif fun==1:#humidity
        try:
            date = datetime.strptime(data['current']['last_updated'], "%Y-%m-%d %H:%M")
            date_num = int(int(date.strftime('%Y%m%d%H%M')) % 100000000) / 10000
            return [int(date_num),data['current']['temp_c'],data['current']['precip_mm'],data['current']['pressure_mb'],data['current']['wind_kph']]  # 与模型训练特征顺序一致
        except KeyError:
            raise Exception("Weather data extraction failed")

def predict_weather(features,fun):
    #load the model
    if fun==0:
        max_temp_topred = scaler_X.transform([features])
        max_temp_scaled = max_temp_model.predict(max_temp_topred)
        max_temp_pred = scaler_y.inverse_transform(max_temp_scaled)[0]

        min_temp_topred = scaler_X2.transform([features])
        min_temp_scaled = min_temp_model.predict(min_temp_topred)
        min_temp_pred = scaler_y2.inverse_transform(min_temp_scaled)[0]

        #save prediction results
        predictions = []
        for day in range(3):
            predictions.append([max_temp_pred[day],min_temp_pred[day]])
        print(predictions)
        return predictions
    elif fun==1:
        humidity_topred = scaler_X3.transform([features])
        humidity_scaled = humidity_model.predict(humidity_topred)
        humidity_pred = scaler_y3.inverse_transform(humidity_scaled)[0]
        predictions = []
        for day in range(3):
            predictions.append(humidity_pred[day])
        return predictions
def query_weather(fun):
    try:
        features = get_weather('London', API_KEY,fun)
        predicted_data = predict_weather(features,fun)
        print(predicted_data)
        return predicted_data
    except Exception as e:
        print("ERROR:", e)
def get_result_temp(query_data):
    if data.find('in three days') != -1:
        pred_temp_data=query_weather(0)
        if data.find('maximum') != -1:
            query_data = "The maximum temperature in three days will be " +f"{pred_temp_data[2][0]:.2f}"+"°C"
        elif data.find('minimum') != -1:
            query_data = "The minimum temperature in three days will be " +f"{pred_temp_data[2][1]:.2f}"+"°C"
        else:
            query_data = "ERROR: Please enter the content you want to predict correctly!"
    elif data.find('the day after tomorrow') != -1:
        pred_temp_data=query_weather(0)
        if data.find('maximum') != -1:
            query_data = "The day after tomorrow's maximum temperature will be " +f"{pred_temp_data[1][0]:.2f}"+"°C"
        elif data.find('minimum') != -1:
            query_data = "The day after tomorrow's minimum temperature will be " +f"{pred_temp_data[1][1]:.2f}"+"°C"
        else:
            query_data = "ERROR: Please enter the content you want to predict correctly!"
    elif data.find('tomorrow') != -1:
        pred_temp_data=query_weather(0)
        if data.find('maximum') != -1:
            query_data = "Tomorrow's maximum temperature will be " +f"{pred_temp_data[0][0]:.2f}"+"°C"
        elif data.find('minimum') != -1:
            query_data = "Tomorrow's minimum temperature will be " +f"{pred_temp_data[0][1]:.2f}"+"°C"
        else:
            query_data = "ERROR: Please enter the content you want to predict correctly!"
    else:
        query_data = "ERROR: Please enter the date correctly!"
    return query_data
def get_result_humidity(query_data):
    humidity_data = query_weather(1)
    if data.find('in three days') != -1:
        query_data = "The humidity in three days will be " + f"{humidity_data[2]:.2f}%"
    elif data.find('the day after tomorrow') != -1:
        query_data = "The day after tomorrow's humidity will be " + f"{humidity_data[1]:.2f}%"
    elif data.find('tomorrow') != -1:
        query_data = "Tomorrow's humidity will be " + f"{humidity_data[0]:.2f}%"
    else:
        query_data = "ERROR: Please enter the date correctly!"
    return query_data
if __name__ == "__main__":
    API_KEY = "cf58533792b7446ebcd191759252403"
    try:
        s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print('Socket created')
    except socket.error as msg:
        print('Failed to create socket. Error Code :' +str(msg[0])+ 'Message' + msg[1])
        sys.exit()
    # Bind socket to local host and port
    try:
        s.bind((HOST,PORT))
    except socket.error as msg:
        print('Bind failed. Error Code : '+str(msg[0]) + 'Message' +msg[1])
        sys.exit()
    print('Socket bind complete')

    # now keep talking with the client
    while 1:
        # receive data from client (data, addr)
        d = s.recvfrom(1024)
        data=d[0].decode()
        #print input
        #print(data)
        # querying for tomorrow's weather
        if data=='Connection successfully established!':
            # send the greeting message(only once)
            greeting = 'Hello, I’m the Oracle. Today is '+ datetime.today().strftime('%Y-%m-%d')+'. How can I help you today?'
            s.sendto(greeting.encode(), d[1])
        else:
            if data.find('temperature') != -1:
                query_data = get_result_temp(data)
            elif data.find('humidity') != -1:
                query_data = get_result_humidity(data)
            else:
                query_data = "ERROR: Please enter the content correctly!"
            addr = d[1]#d[1] : ('127.0.0.1', 53232)
            if not data:
                break
            reply =query_data
            s.sendto(reply.encode(), addr)#send to client
            #print(data.strip())
    s.close()