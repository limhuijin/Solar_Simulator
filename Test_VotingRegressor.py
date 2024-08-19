import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer

# 모델과 스케일러 로드
best_model_percent = joblib.load('C:/Users/user/Desktop/coding/Solar_Simulator/model/model_XGB/model_VotingRegressor_2.pkl')
scaler_X_percent = joblib.load('C:/Users/user/Desktop/coding/Solar_Simulator/model/model_XGB/scaler_X_VotingRegressor_2.gz')
scaler_y_percent = joblib.load('C:/Users/user/Desktop/coding/Solar_Simulator/model/model_XGB/scaler_Y_VotingRegressor_2.gz')

# 데이터 로드 및 전처리
def load_and_preprocess_data(rain_file, temp_file):
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

    # 결측치 처리: 결측값을 평균으로 대체
    imputer = SimpleImputer(strategy='mean')
    merged_df_imputed = pd.DataFrame(imputer.fit_transform(merged_df), columns=merged_df.columns, index=merged_df.index)

    return merged_df_imputed

# 데이터 로드
rain_file = 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/날씨 데이터_1975_2022/1975_2022_기상자료_구미_강수량.csv'
temp_file = 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/날씨 데이터_1975_2022/1975_2022_기상자료_구미_기온.csv'

weather_data = load_and_preprocess_data(rain_file, temp_file)

# 백분율 예측
features_scaled_percent = scaler_X_percent.transform(weather_data)
y_pred_scaled_percent = best_model_percent.predict(features_scaled_percent)
y_pred_percent = scaler_y_percent.inverse_transform(y_pred_scaled_percent.reshape(-1, 1))

# 최대 발전량(kWh) 설정 (예: 1000 kWh)
max_generation_capacity = 9000  # 적절한 값으로 설정해야 합니다

# 예측된 백분율을 이용한 실제 발전량 총량 계산
y_pred_total = (y_pred_percent / 100) * max_generation_capacity

# 예측 결과를 데이터프레임으로 변환
forecast_df_total = pd.DataFrame(y_pred_total, columns=['예측 총량(kWh)'], index=weather_data.index)

# 예측 결과를 CSV 파일로 저장
output_file_path = 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/태양광 데이터/1975_2022_태양광데이터_예측_구미_총량.csv'
forecast_df_total.to_csv(output_file_path, encoding='utf-8-sig')

print(f"예측 결과가 {output_file_path}에 저장되었습니다.")
