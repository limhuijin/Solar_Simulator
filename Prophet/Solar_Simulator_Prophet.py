import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# 경로 설정
samcheonpo_data_path = 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/날씨 데이터/2021_기상자료_예천_통합.csv'

# 기상 데이터 로드 및 전처리
weather_data = pd.read_csv(samcheonpo_data_path)
weather_data['일시'] = pd.to_datetime(weather_data['일시'])
weather_data.set_index('일시', inplace=True)
weather_data = weather_data[['강수량(mm)', '평균기온(℃)', '최고기온(℃)', '최저기온(℃)', '평균풍속(m/s)']]
weather_data = weather_data.sort_index()

# 결측치 처리
weather_data = weather_data.interpolate()
weather_data = weather_data.dropna()

# 저장된 스케일러 로드
scaler_X = joblib.load('C:/Users/user/Desktop/coding/Solar_Simulator/model/scaler_X.gz')
scaler_y = joblib.load('C:/Users/user/Desktop/coding/Solar_Simulator/model/scaler_y.gz')

# 기상 데이터 스케일링
features = weather_data[['강수량(mm)', '평균기온(℃)', '최고기온(℃)', '최저기온(℃)', '평균풍속(m/s)']]
features_scaled = scaler_X.transform(features)

# 저장된 모델 로드
model = XGBRegressor()
model.load_model('C:/Users/user/Desktop/coding/Solar_Simulator/model/model_xgb.json')

# 예측 수행
y_pred_scaled = model.predict(features_scaled)

# 역스케일링
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

# 예측 결과를 데이터프레임으로 변환
forecast_df = pd.DataFrame(y_pred, columns=['총량'], index=weather_data.index)

# 최대 값으로 백분율 계산
max_value = forecast_df['총량'].max()
forecast_df['백분율'] = (forecast_df['총량'] / max_value) * 100

# 시각화 함수 정의
def plot_predictions(weather_data, forecast_df):
    # 날짜 인덱스 추가
    forecast_df['날짜'] = forecast_df.index

    # 시간 기준에 따른 시각화
    time_frames = [0, 1, 2]
    file_names = ['model_365.png', 'model_12.png', 'model_24.png']

    for time_frame, file_name in zip(time_frames, file_names):
        plt.figure(figsize=(12, 6))

        if time_frame == 0:
            # 365일 기준
            daily_predictions = forecast_df.groupby(forecast_df.index.dayofyear)['백분율'].mean()
            days = np.arange(1, 366)
            
            # 일수를 맞추기 위해 빈 값을 0으로 채워서 길이를 맞춤
            if len(daily_predictions) < 365:
                daily_predictions = daily_predictions.reindex(days, fill_value=0)

            plt.plot(days, daily_predictions, label='Predicted Efficiency (%)', color='b')
            plt.xlabel('Day of the Year')
        elif time_frame == 1:
            # 12개월 기준
            monthly_predictions = forecast_df.groupby(forecast_df.index.month)['백분율'].mean()
            months = np.arange(1, 13)
            plt.plot(months, monthly_predictions, label='Predicted Efficiency (%)', color='b')
            plt.xlabel('Month')
        elif time_frame == 2:
            # 24기간 기준 (1년을 24개로 나눔, 각 기간은 약 15일)
            forecast_df['기간'] = (forecast_df.index.dayofyear - 1) // 15
            forecast_df = forecast_df[forecast_df['기간'] < 24]  # 정확히 24개 기간으로 제한
            biweekly_predictions = forecast_df.groupby('기간')['백분율'].mean()
            periods = np.arange(0, 24)  # 0부터 23까지
            
            # 기간수를 맞추기 위해 빈 값을 0으로 채워서 길이를 맞춤
            if len(biweekly_predictions) < 24:
                biweekly_predictions = biweekly_predictions.reindex(periods, fill_value=0)

            plt.plot(periods, biweekly_predictions, label='Predicted Efficiency (%)', color='b')
            plt.xlabel('Biweekly Period')

        # 시각화 및 이미지 저장
        plt.legend()
        plt.title('Predicted Solar Energy Efficiency')
        plt.ylabel('Efficiency (%)')
        plt.grid(True)
        plt.savefig(file_name)
        plt.close()

# 예측 결과 시각화
plot_predictions(weather_data, forecast_df)
