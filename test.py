import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 학습된 모델 로드
model_path = 'C:/Users/user/Desktop/coding/Solar_Simulator/model/model13.keras'
model = load_model(model_path)

# 데이터 파일 경로
file_path = 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/날씨 데이터/2023_기상자료_서울_통합.csv'

# 데이터 로드
data = pd.read_csv(file_path, encoding='utf-8', delimiter=',')

# '일시' 열을 연, 월, 일로 변환
data['일시'] = pd.to_datetime(data['일시'], errors='coerce')
data['연'] = data['일시'].dt.year
data['월'] = data['일시'].dt.month
data['일'] = data['일시'].dt.day

# 결측값 처리
data = data.fillna(0)

# 특성 데이터 설정 (예측에 사용할 변수들)
features = ['강수량(mm)', '1시간최다강수량(mm)', '평균기온(℃)', '최고기온(℃)', '최저기온(℃)', 
            '평균풍속(m/s)', '최대풍속(m/s)', '최대풍속풍향(deg)', '최대순간풍속(m/s)', 
            '최대순간풍속풍향(deg)', '평균습도(%rh)', '최저습도(%rh)']

X = data[features]

# 데이터 정규화
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

# LSTM 입력 형식에 맞게 데이터 변환
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# 예측 수행
predictions_scaled = model.predict(X_scaled)
predictions_scaled = np.maximum(predictions_scaled, 0)  # 음수 값을 0으로 변환
predictions = predictions_scaled.flatten()

# 최대 발전량 계산
max_prediction = np.max(predictions)

# 예측 값을 효율(%)로 변환
if max_prediction != 0:
    efficiency = (predictions / max_prediction) * 100
else:
    efficiency = np.zeros_like(predictions)  # 최대 발전량이 0인 경우 예측 효율을 0으로 설정

# 예측 결과를 날짜별로 그룹화하여 평균값 계산
data['predictions'] = efficiency
data['날짜'] = data['연'].astype(str) + '-' + data['월'].astype(str).str.zfill(2) + '-' + data['일'].astype(str).str.zfill(2)
data['날짜'] = pd.to_datetime(data['날짜'])

# 변수 설정 및 시각화 함수
def plot_and_save(data, time_frame, save_path):
    if time_frame == 0:
        # 365일 기준
        daily_predictions = data.groupby(data['날짜'].dt.dayofyear).mean()['predictions']
        days = np.arange(1, 366)
        plt.figure(figsize=(12, 6))
        plt.plot(days, daily_predictions, label='Predicted Efficiency (%)', color='b')
        plt.xlabel('Day of the Year')
    elif time_frame == 1:
        # 12개월 기준
        monthly_predictions = data.groupby(data['날짜'].dt.month).mean()['predictions']
        months = np.arange(1, 13)
        plt.figure(figsize=(12, 6))
        plt.plot(months, monthly_predictions, label='Predicted Efficiency (%)', color='b')
        plt.xlabel('Month')
    elif time_frame == 2:
        # 24기간 기준 (1년을 24개로 나눔, 각 기간은 약 15일)
        data['기간'] = (data['날짜'].dt.dayofyear - 1) // 15
        data = data[data['기간'] < 24]  # 정확히 24개 기간으로 제한
        biweekly_predictions = data.groupby('기간').mean()['predictions']
        periods = np.arange(0, 24)  # 0부터 23까지
        plt.figure(figsize=(12, 6))
        plt.plot(periods, biweekly_predictions, label='Predicted Efficiency (%)', color='b')
        plt.xlabel('Biweekly Period')

    # 시각화 및 저장
    plt.legend()
    plt.title('Predicted Solar Energy Efficiency')
    plt.ylabel('Predicted Efficiency (%)')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# 시각화 및 이미지 저장
plot_and_save(data, 0, 'model_365_.png')
plot_and_save(data, 1, 'model_12_.png')
plot_and_save(data, 2, 'model_24_.png')
