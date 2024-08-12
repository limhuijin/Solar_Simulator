import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt

# 모델과 스케일러 로드
model = load_model('C:/Users/user/Desktop/coding/Solar_Simulator/model/model_Weather_Forecaster.keras')
scaler = joblib.load('C:/Users/user/Desktop/coding/Solar_Simulator/model/scaler_Weather_Forecaster.pkl')

# 데이터 로드 및 전처리
def load_and_prepare_data(rain_file, temp_file):
    # CSV 파일 로드
    rain_df = pd.read_csv(rain_file)
    temp_df = pd.read_csv(temp_file)

    # '일시'를 datetime 형식으로 변환
    rain_df['일시'] = pd.to_datetime(rain_df['일시'])
    temp_df['일시'] = pd.to_datetime(temp_df['일시'])

    # '일시'를 인덱스로 설정
    rain_df.set_index('일시', inplace=True)
    temp_df.set_index('일시', inplace=True)

    # 필요한 피처만 선택 (학습 시 사용된 피처와 동일하게)
    features_to_use = ['강수량(mm)', '평균기온(℃)', '최고기온(℃)', '최저기온(℃)']

    rain_df = rain_df[['강수량(mm)']]  # 강수량 피처만 사용
    temp_df = temp_df[['평균기온(℃)', '최고기온(℃)', '최저기온(℃)']]  # 기온 관련 피처만 사용

    # 데이터 병합 (inner join으로 날짜 기준 병합)
    merged_df = rain_df.join(temp_df, how='inner')

    # 결측치 처리: 결측값을 0으로 대체
    merged_df = merged_df.fillna(0)

    # 데이터 스케일링 (학습 시 사용된 scaler 사용)
    scaled_data = scaler.transform(merged_df)

    return scaled_data

# 1년치 데이터를 로드
rain_file = 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/날씨 데이터/2022_기상자료_서울_강수량.csv'
temp_file = 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/날씨 데이터/2022_기상자료_서울_기온.csv'

scaled_data = load_and_prepare_data(rain_file, temp_file)

# 시계열 데이터로 변환 (1년치 데이터 사용)
sequence_length = len(scaled_data)
X_input = scaled_data.reshape(1, sequence_length, scaled_data.shape[1])

# 2024년도 날씨 예측
predictions = model.predict(X_input)

# 스케일링 복원 (원래의 스케일로 되돌림)
predictions_reshaped = predictions[0].reshape(-1, scaled_data.shape[1])
predicted_weather_2024 = scaler.inverse_transform(predictions_reshaped)

# 예측 결과 확인
predicted_weather_2024_df = pd.DataFrame(predicted_weather_2024, columns=['강수량(mm)', '평균기온(℃)', '최고기온(℃)', '최저기온(℃)'])

# 강수량 시각화 (강수량 예측이 0 이상이면 비가 온 것으로 처리)
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(predicted_weather_2024_df['강수량(mm)'].apply(lambda x: 1 if x > 0 else 0), label='강수량 예측 (비가 오는 날)')
plt.ylabel('강수량 (1=비가 옴, 0=비가 안 옴)')
plt.title('2024년도 강수량 예측')
plt.legend()

# 기온 시각화
plt.subplot(2, 1, 2)
plt.plot(predicted_weather_2024_df['평균기온(℃)'], label='기온 예측', color='orange')
plt.ylabel('기온 (℃)')
plt.title('2024년도 기온 예측')
plt.legend()

plt.tight_layout()
plt.show()
