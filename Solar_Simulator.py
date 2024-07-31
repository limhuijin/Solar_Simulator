import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 데이터 로드 및 결합
file_paths = [
    'C:/Users/user/Desktop/coding/Solar_Simulator/csv/최종 데이터/merged_solar_weather_data_daily_강원.csv',
    'C:/Users/user/Desktop/coding/Solar_Simulator/csv/최종 데이터/merged_solar_weather_data_daily_구미.csv',
    'C:/Users/user/Desktop/coding/Solar_Simulator/csv/최종 데이터/merged_solar_weather_data_daily_영흥.csv',
    'C:/Users/user/Desktop/coding/Solar_Simulator/csv/최종 데이터/merged_solar_weather_data_daily_진주.csv',
    'C:/Users/user/Desktop/coding/Solar_Simulator/csv/최종 데이터/merged_solar_weather_data_daily_창원.csv'
]

# 데이터 로드 및 결합
data_list = []
for file_path in file_paths:
    try:
        data = pd.read_csv(file_path, encoding='euc-kr')
    except UnicodeDecodeError:
        data = pd.read_csv(file_path, encoding='cp949')
    data_list.append(data)

# 데이터 결합
data = pd.concat(data_list, ignore_index=True)

# '일시' 열을 연, 월, 일로 변환
data['일시'] = pd.to_datetime(data['일시'], errors='coerce')
data['연'] = data['일시'].dt.year
data['월'] = data['일시'].dt.month
data['일'] = data['일시'].dt.day
data = data.drop(columns=['일시'])

# 결측값 처리
data = data.fillna(0)

# 각 지역의 데이터를 평균내어 365일로 압축
data_grouped = data.groupby(['연', '월', '일']).mean().reset_index()

# 특성과 목표 변수 설정
X = data_grouped.drop(columns=['총량'])  # 입력 특성
y = data_grouped['총량']  # 목표 변수

# 데이터 정규화
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# LSTM 입력 형식에 맞게 데이터 변환
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# LSTM 모델 생성
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 학습
history = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.2)

# 모델 평가
loss = model.evaluate(X_test, y_test)
print(f'Model Loss: {loss}')

# 예측
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# 모델 저장
model.save('C:/Users/user/Desktop/coding/Solar_Simulator/model/model.keras')

# 결과 시각화
y_test_actual = scaler_y.inverse_transform(y_test)

plt.figure(figsize=(10, 5))
plt.plot(y_test_actual, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title('Predicted vs Actual Solar Energy Over 365 Days')
plt.xlabel('Day')
plt.ylabel('Solar Energy (kWh)')
plt.show()
