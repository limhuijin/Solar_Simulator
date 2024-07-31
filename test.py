import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 학습된 모델 로드
model_path = 'C:/Users/user/Desktop/coding/Solar_Simulator/model/model4.keras'
model = load_model(model_path)

# 데이터 로드 (예측을 위한 입력 데이터 준비)
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

# 특성과 목표 변수 설정
X = data.drop(columns=['총량'])  # 입력 특성
y = data['총량']  # 목표 변수

# 데이터 정규화
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# LSTM 입력 형식에 맞게 데이터 변환
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# 예측 수행
predictions_scaled = model.predict(X_scaled)
predictions = scaler_y.inverse_transform(predictions_scaled)

# 예측 결과를 날짜별로 그룹화하여 평균값 계산 (365일 기준)
data['predictions'] = predictions
data['날짜'] = data['연'].astype(str) + '-' + data['월'].astype(str).str.zfill(2) + '-' + data['일'].astype(str).str.zfill(2)
data['날짜'] = pd.to_datetime(data['날짜'])
daily_predictions = data.groupby(data['날짜'].dt.dayofyear).mean()['predictions']

# 날짜 설정 (1년 365일)
days = np.arange(1, 366)

# 예측된 효율을 백분율로 변환
efficiency = (daily_predictions / np.max(daily_predictions)) * 100

# 시각화
plt.figure(figsize=(12, 6))
plt.plot(days, efficiency, label='Predicted Efficiency (%)', color='b')
plt.legend()
plt.title('Predicted Solar Energy Efficiency Over 1 Year')
plt.xlabel('Day of the Year')
plt.ylabel('Predicted Efficiency (%)')
plt.grid(True)
plt.show()
