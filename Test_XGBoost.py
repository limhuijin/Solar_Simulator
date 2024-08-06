import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler

# 저장된 모델 불러오기
best_model = joblib.load('best_model.pkl')

# 데이터 전처리
def load_and_preprocess_data():
    # 기상 데이터 로드 및 전처리
    weather_data_path = 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/날씨 데이터/2021_기상자료_삼천포_통합.csv'
    weather_data = pd.read_csv(weather_data_path)
    weather_data['일시'] = pd.to_datetime(weather_data['일시'])
    weather_data.set_index('일시', inplace=True)
    weather_data = weather_data[['강수량(mm)', '평균기온(℃)', '최고기온(℃)', '최저기온(℃)', '평균풍속(m/s)']]
    weather_data = weather_data.sort_index()

    # 태양광 데이터 로드 및 전처리
    solar_data_path = 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/태양광 데이터/2021_태양광데이터_한국남동발전_삼천포.csv'
    solar_data = pd.read_csv(solar_data_path)
    solar_data['년월일'] = pd.to_datetime(solar_data['년월일'])
    solar_data.set_index('년월일', inplace=True)
    solar_data = solar_data.sort_index()
    solar_data['총량(%)'] = solar_data['총량'] / solar_data['총량'].max() * 100
    
    return weather_data, solar_data

weather_data, solar_data = load_and_preprocess_data()

# 스케일러 불러오기
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# 데이터 스케일링 (모델 학습 시 사용한 스케일러와 동일하게 적용)
features_scaled = scaler_X.fit_transform(weather_data)
target_scaled = scaler_y.fit_transform(solar_data['총량(%)'].values.reshape(-1, 1))

# 예측 수행
y_pred_scaled = best_model.predict(features_scaled)

# 역스케일링
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

# 예측 결과를 데이터프레임으로 변환
forecast_df = pd.DataFrame(y_pred, columns=['예측 총량(%)'], index=weather_data.index)

# 실제 데이터와 비교 시각화 함수 정의
def plot_comparison(actual, predicted, time_frame='daily'):
    if time_frame == 'daily':
        actual = actual.resample('D').mean(numeric_only=True)
        predicted = predicted.resample('D').mean()
        title = 'Daily Solar Generation'
        xlabel = 'Day of the Year'
    elif time_frame == 'monthly':
        actual = actual.resample('ME').mean(numeric_only=True)
        predicted = predicted.resample('ME').mean()
        title = 'Monthly Solar Generation'
        xlabel = 'Month'
    elif time_frame == 'biweekly':
        actual = actual.resample('2W').mean(numeric_only=True)
        predicted = predicted.resample('2W').mean()
        title = 'Biweekly Solar Generation'
        xlabel = 'Biweekly Period'

    plt.figure(figsize=(12, 6))
    plt.plot(actual.index, actual['총량(%)'], label='Actual Solar Generation', color='blue')
    plt.plot(predicted.index, predicted['예측 총량(%)'], label='Predicted Solar Generation', color='red')
    plt.xlabel(xlabel)
    plt.ylabel('Solar Generation (%)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# 실제 데이터와 예측 데이터 시각화
plot_comparison(solar_data, forecast_df, 'daily')
plot_comparison(solar_data, forecast_df, 'monthly')
plot_comparison(solar_data, forecast_df, 'biweekly')
