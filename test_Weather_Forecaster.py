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

    # 필요한 피처만 선택
    rain_df = rain_df[['강수량(mm)']]
    temp_df = temp_df[['평균기온(℃)', '최고기온(℃)', '최저기온(℃)']]

    # 데이터 병합 (inner join으로 날짜 기준 병합)
    merged_df = rain_df.join(temp_df, how='inner')

    # 결측치 처리: 결측값을 0으로 대체
    merged_df = merged_df.fillna(0)

    # 데이터 스케일링
    scaled_data = scaler.transform(merged_df)

    return scaled_data, merged_df

# 2022년도 데이터를 로드하여 2023년 예측
rain_file_2022 = 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/날씨 데이터/2021_기상자료_창원_강수량.csv'
temp_file_2022 = 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/날씨 데이터/2021_기상자료_창원_기온.csv'

scaled_data_2022, _ = load_and_prepare_data(rain_file_2022, temp_file_2022)

# 시계열 데이터로 변환 (2022년 데이터를 사용하여 2023년도를 예측)
sequence_length = scaled_data_2022.shape[0]
X_input = scaled_data_2022.reshape(1, sequence_length, scaled_data_2022.shape[1])

# 2023년도 예측
predictions = model.predict(X_input)

# 예측된 데이터를 원래 스케일로 복원
# 1460개의 값을 365일 동안 4개의 변수로 분할
predictions_reshaped = predictions.reshape(365, 4)

predicted_weather_2023 = scaler.inverse_transform(predictions_reshaped)

# 예측 결과를 데이터프레임으로 변환
predicted_weather_2023_df = pd.DataFrame(predicted_weather_2023, columns=["강수량(mm)", "평균기온(℃)", "최고기온(℃)", "최저기온(℃)"])

# 2023년도 실제 데이터를 로드
rain_file_2023 = 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/날씨 데이터/2022_기상자료_창원_강수량.csv'
temp_file_2023 = 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/날씨 데이터/2022_기상자료_창원_기온.csv'

_, actual_data_2023 = load_and_prepare_data(rain_file_2023, temp_file_2023)

# 실제 데이터에서 강수량과 평균기온만 추출
actual_weather_2023_df = actual_data_2023[['강수량(mm)', '평균기온(℃)', '최고기온(℃)', '최저기온(℃)']].reset_index(drop=True)

# 시각화
plt.figure(figsize=(12, 8))

# 강수량 시각화
plt.subplot(2, 1, 1)
plt.plot(predicted_weather_2023_df['강수량(mm)'], label='Predicted Rain')
plt.plot(actual_weather_2023_df['강수량(mm)'], label='Actual Rain', linestyle='--')
plt.ylabel('Rainfall (mm)')
plt.title('Predicted vs Actual Rainfall for 2023')
plt.legend()

# 기온 시각화
plt.subplot(2, 1, 2)
plt.plot(predicted_weather_2023_df['평균기온(℃)'], label='Predicted Temperature', color='orange')
plt.plot(actual_weather_2023_df['평균기온(℃)'], label='Actual Temperature', color='blue', linestyle='--')
plt.ylabel('Temperature (°C)')
plt.title('Predicted vs Actual Temperature for 2023')
plt.legend()

plt.tight_layout()
plt.show()
