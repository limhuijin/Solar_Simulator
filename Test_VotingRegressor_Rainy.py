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
    weather_data = weather_data[['강수량(mm)', '평균기온(℃)', '최고기온(℃)', '최저기온(℃)']]
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
forecast_df_percent['강수량(mm)'] = weather_data['강수량(mm)'].values

# 비가 오는 날과 오지 않는 날로 나눔
rainy_days = forecast_df_percent[forecast_df_percent['강수량(mm)'] > 0]
non_rainy_days = forecast_df_percent[forecast_df_percent['강수량(mm)'] == 0]

# 시각화 함수
def plot_forecast(predicted_data, title):
    plt.figure(figsize=(12, 6))
    plt.plot(predicted_data.index, predicted_data['예측 총량(%)'], label=title, color='red')
    plt.xlabel('Date')
    plt.ylabel('Predicted Solar Generation (%)')
    plt.title(f"{title}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 비가 오는 날의 예측 발전량 그래프
plot_forecast(rainy_days, 'Predicted Solar Generation on Rainy Days')

# 비가 오지 않는 날의 예측 발전량 그래프
plot_forecast(non_rainy_days, 'Predicted Solar Generation on Non-Rainy Days')
