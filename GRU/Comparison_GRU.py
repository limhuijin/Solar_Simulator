import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import joblib

# 데이터 파일 경로 설정
solar_data_path = 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/태양광 데이터/2021_태양광데이터_한국남동발전_삼천포.csv'
weather_data_path = 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/날씨 데이터/2021_기상자료_삼천포_통합.csv'

# 모델 및 스케일러 로드
model = load_model('C:/Users/user/Desktop/coding/Solar_Simulator/model/model.keras')
scaler_X = joblib.load('C:/Users/user/Desktop/coding/Solar_Simulator/model/scaler_X.gz')
scaler_y = joblib.load('C:/Users/user/Desktop/coding/Solar_Simulator/model/scaler_y.gz')

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

# 필요한 특성 설정 및 데이터 정규화
features = ['강수량(mm)', '평균기온(℃)', '최고기온(℃)', '최저기온(℃)', '평균풍속(m/s)', '평균습도(%rh)', '월', '일']
X = weather_data[features]

# 결측값 채우기
X = X.fillna(0)

# 데이터 정규화
X_scaled = scaler_X.transform(X)
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# 예측 수행
predictions_scaled = model.predict(X_scaled)
predictions = scaler_y.inverse_transform(predictions_scaled).flatten()

# 최대 발전량 계산
pred_max = np.max(predictions)

# 예측 값을 효율(%)로 변환
if pred_max != 0:
    efficiency = (predictions / pred_max) * 100
else:
    efficiency = np.zeros_like(predictions)  # 최대 발전량이 0인 경우 예측 효율을 0으로 설정

# 예측 결과를 날짜별로 그룹화하여 평균값 계산
weather_data['예측값'] = efficiency
weather_data['날짜'] = pd.to_datetime(weather_data['연'].astype(str) + '-' + weather_data['월'].astype(str).str.zfill(2) + '-' + weather_data['일'].astype(str).str.zfill(2))

# 실제 데이터와 예측 데이터 합치기
solar_data['년월일'] = solar_data['년월일'].dt.floor('D')
weather_data['날짜'] = weather_data['날짜'].dt.floor('D')

merged_data = pd.merge(solar_data[['년월일', '효율']], weather_data[['날짜', '예측값']], left_on='년월일', right_on='날짜', how='inner')

# 시각화 준비
segments = 24
segment_size = len(merged_data) // segments

actual_segmented = [np.mean(merged_data['효율'][i*segment_size:(i+1)*segment_size]) for i in range(segments)]
predicted_segmented = [np.mean(merged_data['예측값'][i*segment_size:(i+1)*segment_size]) for i in range(segments)]

plt.figure(figsize=(14, 7))
plt.plot(actual_segmented, label='Actual Solar Efficiency (24 segments)', color='blue')
plt.plot(predicted_segmented, label='Predicted Solar Efficiency (24 segments)', color='red', linestyle='--')
plt.xlabel('Segment')
plt.ylabel('Efficiency (%)')
plt.title('Comparison of Actual and Predicted Solar Efficiency (24 segments)')
plt.legend()
plt.grid(True)
plt.show()

# 전체 오차율(MAPE) 계산 및 출력
non_zero_indices = merged_data['효율'] != 0
mape = np.mean(np.abs((merged_data['효율'][non_zero_indices] - merged_data['예측값'][non_zero_indices]) / merged_data['효율'][non_zero_indices])) * 100
print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')
