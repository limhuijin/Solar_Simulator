import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# 지역별 데이터 병합 함수
def load_and_merge_data_for_region(region_name, base_path):
    rain_file = os.path.join(base_path, f'1975_2022_기상자료_{region_name}_강수량.csv')
    temp_file = os.path.join(base_path, f'1975_2022_기상자료_{region_name}_기온.csv')

    # CSV 파일 로드
    rain_df = pd.read_csv(rain_file)
    temp_df = pd.read_csv(temp_file)

    # '일시'를 datetime 형식으로 변환
    rain_df['일시'] = pd.to_datetime(rain_df['일시'])
    temp_df['일시'] = pd.to_datetime(temp_df['일시'])

    # '일시'를 인덱스로 설정
    rain_df.set_index('일시', inplace=True)
    temp_df.set_index('일시', inplace=True)

    # 강수량과 기온 데이터에서 필요한 피처만 선택
    rain_df = rain_df[['강수량(mm)']]
    temp_df = temp_df[['평균기온(℃)', '최고기온(℃)', '최저기온(℃)']]

    # 데이터 병합 (inner join으로 날짜 기준 병합)
    merged_df = rain_df.join(temp_df, how='inner')

    # 결측치 처리: 결측값을 0으로 대체
    merged_df = merged_df.fillna(0)

    return merged_df

# 데이터 로드 및 병합
base_path = 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/날씨 데이터_1975_2022/'  # 모든 CSV 파일이 위치한 기본 경로
regions = ["구미", "영흥", "예천", "진주"]  # 창원을 제외
all_sequences = []

for region in regions:
    merged_data = load_and_merge_data_for_region(region, base_path)

    # 데이터 스케일링
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(merged_data)

    # 시계열 데이터 생성 (47년 데이터를 포함하는 시퀀스)
    sequence_length = 365 * 47  # 47년의 데이터를 포함하는 시퀀스
    X, y = [], []

    for i in range(sequence_length, len(scaled_data) - 365):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i:i+365, :])  # 365일 동안의 모든 피처 예측

    # 지역별 데이터를 추가
    all_sequences.append((np.array(X), np.array(y)))

# 모든 지역에서 생성된 시퀀스를 결합
X_train = np.concatenate([seq[0] for seq in all_sequences], axis=0)
y_train = np.concatenate([seq[1] for seq in all_sequences], axis=0)

# X_train과 y_train의 형태 확인
print(f"X_train.shape: {X_train.shape}")
print(f"y_train.shape: {y_train.shape}")

# LSTM 모델 생성
model = Sequential()

# LSTM 레이어 추가
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.2))

# 출력 레이어 (365일 동안의 모든 피처 예측)
model.add(Dense(units=365 * 4))  # 365일 동안의 4가지 피처(강수량, 평균기온, 최고기온, 최저기온)를 예측

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 훈련
history = model.fit(X_train, y_train.reshape(y_train.shape[0], -1), epochs=50, batch_size=32)

# 모델을 .keras 형식으로 저장
model.save('C:/Users/user/Desktop/coding/Solar_Simulator/model/model_Weather_Forecaster.keras')

# 스케일러 저장
import joblib
joblib.dump(scaler, 'C:/Users/user/Desktop/coding/Solar_Simulator/model/scaler_Weather_Forecaster.pkl')
