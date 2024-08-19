import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 백분율 예측 모델 불러오기
best_model_percent = joblib.load('C:/Users/user/Desktop/coding/Solar_Simulator/model/model_XGB/model_VotingRegressor_3.pkl')
scaler_X_percent = joblib.load('C:/Users/user/Desktop/coding/Solar_Simulator/model/model_XGB/scaler_X_VotingRegressor_3.gz')
scaler_y_percent = joblib.load('C:/Users/user/Desktop/coding/Solar_Simulator/model/model_XGB/scaler_Y_VotingRegressor_3.gz')

# 데이터 전처리
def load_and_preprocess_data():
    weather_data_path = 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/날씨 데이터/2021_기상자료_예천_통합.csv'
    weather_data = pd.read_csv(weather_data_path)
    weather_data['일시'] = pd.to_datetime(weather_data['일시'])
    weather_data.set_index('일시', inplace=True)
    weather_data = weather_data[['강수량(mm)', '평균기온(℃)', '최고기온(℃)', '최저기온(℃)']]
    weather_data = weather_data.sort_index()

    solar_data_path = 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/태양광 데이터/2021_태양광데이터_한국남동발전_예천.csv'
    solar_data = pd.read_csv(solar_data_path)
    solar_data['년월일'] = pd.to_datetime(solar_data['년월일'])
    solar_data.set_index('년월일', inplace=True)
    solar_data = solar_data.sort_index()
    solar_data['총량(%)'] = solar_data['총량'] / solar_data['총량'].max() * 100
    
    return weather_data, solar_data

weather_data, solar_data = load_and_preprocess_data()

# NaN 값 처리
imputer = SimpleImputer(strategy='mean')
weather_data_imputed = imputer.fit_transform(weather_data)

# 백분율 예측
features_scaled_percent = scaler_X_percent.transform(weather_data_imputed)
y_pred_scaled_percent = best_model_percent.predict(features_scaled_percent)
y_pred_percent = scaler_y_percent.inverse_transform(y_pred_scaled_percent.reshape(-1, 1))

# 최대 발전량(kWh) 계산
max_generation_capacity = solar_data['총량'].max()

# 예측된 백분율을 이용한 발전량 계산
predicted_generation_kwh = (y_pred_percent / 100) * max_generation_capacity

# 예측 결과를 데이터프레임으로 변환
forecast_df_percent = pd.DataFrame(y_pred_percent, columns=['예측 총량(%)'], index=weather_data.index)
forecast_df_actual = pd.DataFrame(predicted_generation_kwh, columns=['예측 총량(kWh)'], index=weather_data.index)

# 실제 값과의 비교를 위해 RMSE, MAE 계산
mae_percent = mean_absolute_error(solar_data['총량(%)'], forecast_df_percent['예측 총량(%)'])
rmse_percent = mean_squared_error(solar_data['총량(%)'], forecast_df_percent['예측 총량(%)'], squared=False)

mae_actual = mean_absolute_error(solar_data['총량'], forecast_df_actual['예측 총량(kWh)'])
rmse_actual = mean_squared_error(solar_data['총량'], forecast_df_actual['예측 총량(kWh)'], squared=False)

print(f'Mean Absolute Error (MAE) - Percent: {mae_percent:.2f}%')
print(f'Root Mean Squared Error (RMSE) - Percent: {rmse_percent:.2f}%')
print(f'Mean Absolute Error (MAE) - Actual: {mae_actual:.2f}')
print(f'Root Mean Squared Error (RMSE) - Actual: {rmse_actual:.2f}')

# 총 발전량 계산 함수
def calculate_totals(predicted_actual):
    daily_total = predicted_actual.resample('D').sum().mean().values[0]
    biweekly_total = predicted_actual.resample('2W').sum().mean().values[0]
    monthly_total = predicted_actual.resample('ME').sum().mean().values[0]
    yearly_total = predicted_actual.resample('YE').sum().values[0]
    return daily_total, biweekly_total, monthly_total, yearly_total

# 시각화 함수
def plot_comparison(actual, predicted_percent, predicted_actual, time_frame='daily'):
    daily_total, biweekly_total, monthly_total, yearly_total = calculate_totals(predicted_actual)
    
    if time_frame == 'daily':
        actual_percent = actual.resample('D').mean(numeric_only=True)
        predicted_percent = predicted_percent.resample('D').mean()
        title = 'Daily Solar Generation'
        xlabel = 'Day of the Year'
        total_text = f"Daily Total: {daily_total:.2f} kWh"
    elif time_frame == 'monthly':
        actual_percent = actual.resample('ME').mean(numeric_only=True)  # 월평균 계산
        predicted_percent = predicted_percent.resample('ME').mean()
        title = 'Monthly Solar Generation'
        xlabel = 'Month'
        total_text = f"Monthly Total: {monthly_total:.2f} kWh"
    elif time_frame == 'biweekly':
        actual_percent = actual.resample('2W').mean(numeric_only=True)
        predicted_percent = predicted_percent.resample('2W').mean()
        title = 'Biweekly Solar Generation'
        xlabel = 'Biweekly Period'
        total_text = f"Biweekly Total: {biweekly_total:.2f} kWh"

    yearly_text = f"Yearly Total: {yearly_total[0]:.2f} kWh"
    mae_text = f"MAE (Percent): {mae_percent:.2f}%\n{total_text} | {yearly_text}"

    plt.figure(figsize=(12, 8))
    
    plt.plot(actual_percent.index, actual_percent['총량(%)'], label='Actual Solar Generation (%)', color='blue')
    plt.plot(predicted_percent.index, predicted_percent['예측 총량(%)'], label='Predicted Solar Generation (%)', color='red')
    plt.xlabel(xlabel)
    plt.ylabel('Solar Generation (%)')
    plt.title(f"{title} (Percentage)\n{mae_text}")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# 실제 데이터와 예측 데이터 시각화
plot_comparison(solar_data, forecast_df_percent, forecast_df_actual, 'daily')
plot_comparison(solar_data, forecast_df_percent, forecast_df_actual, 'monthly')
plot_comparison(solar_data, forecast_df_percent, forecast_df_actual, 'biweekly')

# 예측 결과를 CSV 파일로 저장
forecast_df_actual.to_csv('C:/Users/user/Desktop/coding/Solar_Simulator/predicted_solar_generation_01.csv', encoding='utf-8-sig')

print("Predicted solar generation data has been saved to 'predicted_solar_generation_01.csv'.")
