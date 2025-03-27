import csv
#import chardet
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

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)


# 构建全连接神经网络
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
# model=load_model("min_temp_predictor_model.h5")
# scaler_X = joblib.load('scaler_X2.pkl')
# scaler_y = joblib.load('scaler_y2.pkl')
# X_pred = scaler_X.transform([[326,53,0,1024,9.36]])
# y_pred_scaled = model.predict(X_pred)
# y_pred = scaler_y.inverse_transform(y_pred_scaled)[0]

# print("未来7天气温预测：", y_pred)

'''
data = pd.read_csv('new_data.csv')
data=data.dropna()
readCSV = csv.reader(data, delimiter=',')
# 假设数据包含列 'temperature', 'humidity', 'pressure' 等
# 我们使用这些特征来预测温度（temperature）
X = data[['mean_temp','humidity','precipitation']]  # 特征列
y = data[['sunshine']] # 目标列

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 使用随机森林回归器训练模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 在测试集上评估模型
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# 保存模型

joblib.dump(model, 'weather_predictor_model2.pkl')

def get_yesterday_weather(city, api_key):
    # 获取昨日日期
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    print(datetime.now())
    # 调用 WeatherAPI 获取历史天气
    url = f"https://api.weatherapi.com/v1/history.json"
    params = {'key': api_key, 'q': city, 'dt': yesterday}

    response = requests.get(url, params=params)
    data = response.json()

    try:
        day_data = data['forecast']['forecastday'][0]['day']
        avg_temp = day_data['avgtemp_c']
        humidity = day_data['avghumidity']
        precip = day_data['totalprecip_mm']
        print(day_data)
        print(f"{city} 昨日天气：均温 {avg_temp}℃，湿度 {humidity}%，降水 {precip}mm")

        return [avg_temp, humidity, precip]  # 与模型训练特征顺序一致
    except KeyError:
        raise Exception("天气数据提取失败，请检查响应数据结构")


def predict_tomorrow_weather(features):
    # 加载模型并预测
    model = joblib.load('weather_predictor_model.pkl')

    # 要预测几天？
    days_to_predict = 3

    # 保存每一天的预测结果
    predictions = []

    for day in range(days_to_predict):
        # 预测当前这一天的温度
        print(model.predict(np.array(features).reshape(1, -1))[0])
        predicted_temp = model.predict(np.array(features).reshape(1, -1))[0]
        predicted_temp_recursive=np.mean(model.predict(np.array(features).reshape(1, -1))[0])
        predictions.append(predicted_temp)

        #print(f"第 {day + 1} 天预测温度：{predicted_temp:.2f} °C")

        features = [predicted_temp_recursive, features[1], features[2]]
    return  predictions
    
    X = np.array(features).reshape(1, -1)
    prediction = model.predict(X)
    print(prediction)
    return prediction
    

# === 3. 主程序 ===
if __name__ == "__main__":
    API_KEY = "cf58533792b7446ebcd191759252403"

    try:
        yesterday_features = get_yesterday_weather('London', API_KEY)
        predicted_temp = predict_tomorrow_weather(yesterday_features)
        print(f"\nLondon明天的最高温度为：{predicted_temp[1][0]:.2f} °C，最低温度为：{predicted_temp[1][1]:.2f} °C")
    except Exception as e:
        print("发生错误：", e)
'''