import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# 백분율 예측 모델 불러오기
best_model_percent = joblib.load('C:/Users/user/Desktop/coding/Solar_Simulator/model/model_XGB/model_VotingRegressor_2.pkl')
scaler_X_percent = joblib.load('C:/Users/user/Desktop/coding/Solar_Simulator/model/model_XGB/scaler_X_VotingRegressor_2.gz')
scaler_y_percent = joblib.load('C:/Users/user/Desktop/coding/Solar_Simulator/model/model_XGB/scaler_Y_VotingRegressor_2.gz')

# 데이터 전처리
def load_and_preprocess_data():
    weather_data_path = 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/날씨 데이터/2017_기상자료_예천_통합.csv'
    weather_data = pd.read_csv(weather_data_path)
    weather_data['일시'] = pd.to_datetime(weather_data['일시'])
    weather_data.set_index('일시', inplace=True)
    weather_data = weather_data[['강수량(mm)', '평균기온(℃)', '최고기온(℃)', '최저기온(℃)', '평균풍속(m/s)']]
    weather_data = weather_data.sort_index()
    
    return weather_data

# 기상 데이터만 사용하여 예측 수행
weather_data = load_and_preprocess_data()

# NaN 값 처리
imputer = SimpleImputer(strategy='mean')
weather_data_imputed = imputer.fit_transform(weather_data)

# 백분율 예측
features_scaled_percent = scaler_X_percent.transform(weather_data_imputed)
y_pred_scaled_percent = best_model_percent.predict(features_scaled_percent)
y_pred_percent = scaler_y_percent.inverse_transform(y_pred_scaled_percent.reshape(-1, 1))

# 예측 결과를 데이터프레임으로 변환
forecast_df_percent = pd.DataFrame(y_pred_percent, columns=['예측 총량(%)'], index=weather_data.index)

# 시각화 함수
def plot_forecast(predicted_percent, time_frame='daily'):
    if time_frame == 'daily':
        predicted_percent = predicted_percent.resample('D').mean()
        title = 'Daily Solar Generation Prediction'
        xlabel = 'Day of the Year'
    elif time_frame == 'monthly':
        predicted_percent = predicted_percent.resample('ME').mean()
        title = 'Monthly Solar Generation Prediction'
        xlabel = 'Month'
    elif time_frame == 'biweekly':
        predicted_percent = predicted_percent.resample('2W').mean()
        title = 'Biweekly Solar Generation Prediction'
        xlabel = 'Biweekly Period'

    plt.figure(figsize=(12, 8))
    
    plt.plot(predicted_percent.index, predicted_percent['예측 총량(%)'], label='Predicted Solar Generation (%)', color='red')
    plt.xlabel(xlabel)
    plt.ylabel('Solar Generation (%)')
    plt.title(f"{title}")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# 예측 데이터 시각화
plot_forecast(forecast_df_percent, 'daily')
plot_forecast(forecast_df_percent, 'monthly')
plot_forecast(forecast_df_percent, 'biweekly')