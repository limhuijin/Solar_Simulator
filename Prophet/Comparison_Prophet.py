import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from prophet import Prophet

# 데이터 파일 경로 설정
solar_data_path = 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/태양광 데이터/2021_태양광데이터_한국남동발전_삼천포.csv'
weather_data_path = 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/날씨 데이터/2021_기상자료_예천_삼천포.csv'

# 모델 로드
model_path = 'C:/Users/user/Desktop/coding/Solar_Simulator/model/model_prophet.pkl'
model = joblib.load(model_path)

# 태양광 데이터 로드 및 전처리
solar_data = pd.read_csv(solar_data_path, encoding='utf-8')
solar_data['년월일'] = pd.to_datetime(solar_data['년월일'], errors='coerce')
solar_data['총량'] = solar_data['총량'].fillna(0)  # 결측값을 0으로 대체
solar_max = solar_data['총량'].max()
solar_data['효율'] = (solar_data['총량'] / solar_max) * 100

# 기상 데이터 로드 및 전처리
weather_data = pd.read_csv(weather_data_path, encoding='utf-8')
weather_data['일시'] = pd.to_datetime(weather_data['일시'], errors='coerce')
weather_data['연'] = weather_data['일시'].dt.year
weather_data['월'] = weather_data['일시'].dt.month
weather_data['일'] = weather_data['일시'].dt.day
weather_data['요일'] = weather_data['일시'].dt.dayofweek

# 결측값 채우기
weather_data['강수량(mm)'] = weather_data['강수량(mm)'].interpolate().fillna(0)
weather_data = weather_data.fillna(0)

# 필요한 특성 설정
features = ['일시', '강수량(mm)', '평균기온(℃)', '최고기온(℃)', '최저기온(℃)', '평균풍속(m/s)', '평균습도(%rh)']
weather_data = weather_data[features]
weather_data = weather_data.rename(columns={'일시': 'ds'})

# 예측 수행
forecast = model.predict(weather_data)

# 최대 발전량 계산
max_prediction = forecast['yhat'].max()

# 예측 값을 효율(%)로 변환
if max_prediction != 0:
    forecast['efficiency'] = (forecast['yhat'] / max_prediction) * 100
else:
    forecast['efficiency'] = np.zeros_like(forecast['yhat'])  # 최대 발전량이 0인 경우 예측 효율을 0으로 설정

# 예측 결과를 날짜별로 그룹화하여 평균값 계산
forecast['날짜'] = forecast['ds'].dt.floor('D')

# 실제 데이터와 예측 데이터 합치기
solar_data['년월일'] = solar_data['년월일'].dt.floor('D')
forecast['날짜'] = forecast['날짜'].astype('datetime64[ns]')

merged_data = pd.merge(solar_data[['년월일', '효율']], forecast[['날짜', 'efficiency']], left_on='년월일', right_on='날짜', how='inner')

# 시각화 준비
segments = 24
segment_size = len(merged_data) // segments

actual_segmented = [np.mean(merged_data['효율'][i*segment_size:(i+1)*segment_size]) for i in range(segments)]
predicted_segmented = [np.mean(merged_data['efficiency'][i*segment_size:(i+1)*segment_size]) for i in range(segments)]

# 시각화
plt.figure(figsize=(14, 7))
plt.plot(actual_segmented, label='Actual Solar Efficiency (24 segments)', color='blue')
plt.plot(predicted_segmented, label='Predicted Solar Efficiency (24 segments)', color='red', linestyle='--')
plt.xlabel('Segment')
plt.ylabel('Efficiency (%)')
plt.title('Comparison of Actual and Predicted Solar Efficiency (24 segments)')
plt.legend()
plt.grid(True)
plt.show()
