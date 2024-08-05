import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt

# 모델 및 스케일러 로드
model = load_model('C:/Users/user/Desktop/coding/Solar_Simulator/model/model.keras')
scaler_X = joblib.load('C:/Users/user/Desktop/coding/Solar_Simulator/model/scaler_X.gz')
scaler_y = joblib.load('C:/Users/user/Desktop/coding/Solar_Simulator/model/scaler_y.gz')

# 기상 데이터 로드 및 전처리
weather_data_path = 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/날씨 데이터/2021_기상자료_삼천포_통합.csv'
weather_data = pd.read_csv(weather_data_path, encoding='utf-8')
weather_data['일시'] = pd.to_datetime(weather_data['일시'], errors='coerce')
weather_data['연'] = weather_data['일시'].dt.year
weather_data['월'] = weather_data['일시'].dt.month
weather_data['일'] = weather_data['일시'].dt.day
weather_data['요일'] = weather_data['일시'].dt.dayofweek

# 필요한 특성 설정 및 데이터 준비
features = ['강수량(mm)', '평균기온(℃)', '최고기온(℃)', '최저기온(℃)', '평균풍속(m/s)', '평균습도(%rh)', '월', '일']
weather_data = weather_data[features]

# 결측값 처리
weather_data['강수량(mm)'] = weather_data['강수량(mm)'].fillna(0)
weather_data = weather_data.fillna(weather_data.mean())

# 데이터 정규화
X_scaled = scaler_X.transform(weather_data)
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# 예측 수행
predictions_scaled = model.predict(X_scaled)
predictions = scaler_y.inverse_transform(predictions_scaled).flatten()

# 최대 발전량 계산
max_prediction = np.max(predictions)

# 예측 값을 효율(%)로 변환
if max_prediction != 0:
    efficiency = (predictions / max_prediction) * 100
else:
    efficiency = np.zeros_like(predictions)  # 최대 발전량이 0인 경우 예측 효율을 0으로 설정

# 예측 결과를 날짜별로 그룹화하여 평균값 계산
weather_data['predictions'] = efficiency
weather_data['날짜'] = pd.date_range(start='2021-01-01', periods=len(weather_data), freq='D')

# 변수 설정 (0: 365일, 1: 12개월, 2: 24기간)
time_frames = [0, 1, 2]
file_names = ['model_365.png', 'model_12.png', 'model_24.png']

for time_frame, file_name in zip(time_frames, file_names):
    if time_frame == 0:
        # 365일 기준
        daily_predictions = weather_data.groupby(weather_data['날짜'].dt.dayofyear)['predictions'].mean()
        days = np.arange(1, 366)
        plt.figure(figsize=(12, 6))
        plt.plot(days, daily_predictions, label='Predicted Efficiency (%)', color='b')
        plt.xlabel('Day of the Year')
    elif time_frame == 1:
        # 12개월 기준
        monthly_predictions = weather_data.groupby(weather_data['날짜'].dt.month)['predictions'].mean()
        months = np.arange(1, 13)
        plt.figure(figsize=(12, 6))
        plt.plot(months, monthly_predictions, label='Predicted Efficiency (%)', color='b')
        plt.xlabel('Month')
    elif time_frame == 2:
        # 24기간 기준 (1년을 24개로 나눔, 각 기간은 약 15일)
        weather_data['기간'] = (weather_data['날짜'].dt.dayofyear - 1) // 15
        weather_data = weather_data[weather_data['기간'] < 24]  # 정확히 24개 기간으로 제한
        biweekly_predictions = weather_data.groupby('기간')['predictions'].mean()
        periods = np.arange(0, 24)  # 0부터 23까지
        plt.figure(figsize=(12, 6))
        plt.plot(periods, biweekly_predictions, label='Predicted Efficiency (%)', color='b')
        plt.xlabel('Biweekly Period')
    
    # 시각화 및 이미지 저장
    plt.legend()
    plt.title('Predicted Solar Energy Efficiency')
    plt.ylabel('Predicted Efficiency (%)')
    plt.grid(True)
    plt.savefig(file_name)
    plt.close()

print("Images saved:")
print(" - model_365.png")
print(" - model_12.png")
print(" - model_24.png")
