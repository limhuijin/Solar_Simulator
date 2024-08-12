import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
import numpy as np

# 경로 설정
samcheonpo_data_path = 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/날씨 데이터/2021_기상자료_삼천포_통합.csv'

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
scaler_features = joblib.load('scaler_features.pkl')
scaler_target = joblib.load('scaler_target.pkl')

# 기상 데이터 스케일링
features = weather_data[['강수량(mm)', '평균기온(℃)', '최고기온(℃)', '최저기온(℃)', '평균풍속(m/s)']]
features_scaled = scaler_features.transform(features)

# 저장된 모델 로드
model = joblib.load('var_model.pkl')

# VAR 모델을 사용해 예측
forecast_input = pd.DataFrame(features_scaled, columns=['강수량(mm)', '평균기온(℃)', '최고기온(℃)', '최저기온(℃)', '평균풍속(m/s)'])
forecast = model.forecast(y=forecast_input.values, steps=len(forecast_input))

# 예측 결과를 데이터프레임으로 변환
forecast_df = pd.DataFrame(forecast, columns=['강수량(mm)', '평균기온(℃)', '최고기온(℃)', '최저기온(℃)', '평균풍속(m/s)'])
forecast_df.index = weather_data.index

# 역스케일링
forecast_df['총량'] = scaler_target.inverse_transform(forecast_df[['총량']].values)

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
            daily_predictions = forecast_df.groupby(forecast_df.index.dayofyear)['총량'].mean()
            days = np.arange(1, 366)
            plt.plot(days, daily_predictions, label='Predicted Total', color='b')
            plt.xlabel('Day of the Year')
        elif time_frame == 1:
            # 12개월 기준
            monthly_predictions = forecast_df.groupby(forecast_df.index.month)['총량'].mean()
            months = np.arange(1, 13)
            plt.plot(months, monthly_predictions, label='Predicted Total', color='b')
            plt.xlabel('Month')
        elif time_frame == 2:
            # 24기간 기준 (1년을 24개로 나눔, 각 기간은 약 15일)
            forecast_df['기간'] = (forecast_df.index.dayofyear - 1) // 15
            forecast_df = forecast_df[forecast_df['기간'] < 24]  # 정확히 24개 기간으로 제한
            biweekly_predictions = forecast_df.groupby('기간')['총량'].mean()
            periods = np.arange(0, 24)  # 0부터 23까지
            plt.plot(periods, biweekly_predictions, label='Predicted Total', color='b')
            plt.xlabel('Biweekly Period')

        # 시각화 및 이미지 저장
        plt.legend()
        plt.title('Predicted Total Solar Energy')
        plt.ylabel('Total Solar Energy')
        plt.grid(True)
        plt.savefig(file_name)
        plt.close()

# 예측 결과 시각화
plot_predictions(weather_data, forecast_df)
