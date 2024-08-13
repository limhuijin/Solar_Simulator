import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error  # 추가된 부분

# 모델과 스케일러 로드
model = load_model('C:/Users/user/Desktop/coding/Solar_Simulator/model/model_Weather_Forecaster_full.keras')
scaler = joblib.load('C:/Users/user/Desktop/coding/Solar_Simulator/model/scaler_Weather_Forecaster_full.pkl')

best_model_percent = joblib.load('C:/Users/user/Desktop/coding/Solar_Simulator/model/model_XGB/model_VotingRegressor_2.pkl')  # 추가된 부분
scaler_X_percent = joblib.load('C:/Users/user/Desktop/coding/Solar_Simulator/model/model_XGB/scaler_X_VotingRegressor_2.gz')  # 추가된 부분
scaler_y_percent = joblib.load('C:/Users/user/Desktop/coding/Solar_Simulator/model/model_XGB/scaler_Y_VotingRegressor_2.gz')  # 추가된 부분

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

# 강수량이 음수로 예측된 경우 0으로 설정 (추가된 부분)
predicted_weather_2023[:, 0] = np.maximum(predicted_weather_2023[:, 0], 0)

# 예측 결과를 데이터프레임으로 변환
predicted_weather_2023_df = pd.DataFrame(predicted_weather_2023, columns=["강수량(mm)", "평균기온(℃)", "최고기온(℃)", "최저기온(℃)"])

# 2023년도 실제 데이터를 로드
rain_file_2023 = 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/날씨 데이터/2022_기상자료_창원_강수량.csv'
temp_file_2023 = 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/날씨 데이터/2022_기상자료_창원_기온.csv'

_, actual_data_2023 = load_and_prepare_data(rain_file_2023, temp_file_2023)

# 실제 데이터에서 강수량과 평균기온만 추출
actual_weather_2023_df = actual_data_2023[['강수량(mm)', '평균기온(℃)', '최고기온(℃)', '최저기온(℃)']].reset_index(drop=True)

# 태양광 발전량 예측을 위해 필요한 피처 선택 및 스케일링 (추가된 부분)
solar_features_predicted = predicted_weather_2023_df[['강수량(mm)', '평균기온(℃)', '최고기온(℃)', '최저기온(℃)']]
solar_features_actual = actual_weather_2023_df[['강수량(mm)', '평균기온(℃)', '최고기온(℃)', '최저기온(℃)']]

solar_features_predicted_scaled = scaler_X_percent.transform(solar_features_predicted)
solar_features_actual_scaled = scaler_X_percent.transform(solar_features_actual)

solar_generation_predicted_scaled = best_model_percent.predict(solar_features_predicted_scaled)
solar_generation_predicted_percent = scaler_y_percent.inverse_transform(solar_generation_predicted_scaled.reshape(-1, 1))

# 최대 발전량(kWh) 계산 (추가된 부분)
solar_actual_path = 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/태양광 데이터/2022_태양광데이터_한국남동발전_창원.csv'
solar_actual_df = pd.read_csv(solar_actual_path)
solar_actual_df['년월일'] = pd.to_datetime(solar_actual_df['년월일'])
solar_actual_df.set_index('년월일', inplace=True)
actual_solar_generation = solar_actual_df['총량'].values[:len(predicted_weather_2023_df)]

max_generation_capacity = solar_actual_df['총량'].max()

# 예측된 백분율을 이용한 발전량 계산 (추가된 부분)
predicted_generation_kwh = (solar_generation_predicted_percent / 100) * max_generation_capacity

# MAE 계산 (추가된 부분)
mae = mean_absolute_error(actual_solar_generation, predicted_generation_kwh)
print(f"MAE: {mae}")

# 시각화
plt.figure(figsize=(18, 24))

# 강수량 비교 시각화
plt.subplot(4, 1, 1)
plt.plot(predicted_weather_2023_df['강수량(mm)'], label='Predicted Rain')
plt.plot(actual_weather_2023_df['강수량(mm)'], label='Actual Rain', linestyle='--')
plt.ylabel('Rainfall (mm)')
plt.title('Predicted vs Actual Rainfall for 2023')
plt.legend()

# 기온 비교 시각화
plt.subplot(4, 1, 2)
plt.plot(predicted_weather_2023_df['평균기온(℃)'], label='Predicted Temperature', color='orange')
plt.plot(actual_weather_2023_df['평균기온(℃)'], label='Actual Temperature', color='blue', linestyle='--')
plt.ylabel('Temperature (°C)')
plt.title('Predicted vs Actual Temperature for 2023')
plt.legend()

# 태양광 발전량 예측 및 실제 발전량 비교 시각화 (백분율) (추가된 부분)
plt.subplot(4, 1, 3)
plt.plot(predicted_weather_2023_df.index, solar_generation_predicted_percent, label='Predicted Solar Generation (%)', color='green')
plt.plot(predicted_weather_2023_df.index, actual_solar_generation / max_generation_capacity * 100, label='Actual Solar Generation (%)', color='red', linestyle=':')
plt.ylabel('Solar Generation (%)')
plt.title(f'Solar Generation: Prediction vs Actual (Percentage)\nMAE: {mae:.2f}')
plt.legend()

plt.tight_layout()
plt.show()
