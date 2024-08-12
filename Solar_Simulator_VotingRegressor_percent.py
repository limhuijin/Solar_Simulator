import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# 모든 CSV 파일을 읽어 병합하는 함수
def load_and_merge_data_from_path(path):
    all_files = glob.glob(os.path.join(path, "*.csv"))
    merged_data = pd.DataFrame()

    for file in all_files:
        df = pd.read_csv(file)
        df['일시'] = pd.to_datetime(df['일시'])  # 일시 열을 datetime 형식으로 변환
        df.set_index('일시', inplace=True)  # 일시를 인덱스로 설정

        # 데이터 병합 (inner join으로 일치하는 인덱스만 병합)
        if merged_data.empty:
            merged_data = df
        else:
            merged_data = merged_data.join(df, how='inner')

    # 결측치 처리 (필요시)
    merged_data = merged_data.fillna(method='ffill')

    return merged_data

# 데이터 로드 및 병합
path_to_files = 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/날씨 데이터_1975_2022'  # 모든 CSV 파일이 위치한 경로
merged_data = load_and_merge_data_from_path(path_to_files)

# 데이터 스케일링
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(merged_data)

# 시계열 데이터 생성 (50년 데이터로 이번 연도 예측)
sequence_length = 365 * 47  # 50년의 데이터
X, y = [], []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i:i+365, :2])  # 365일 예측 (강수량과 평균기온)

X, y = np.array(X), np.array(y)

# 학습 데이터와 테스트 데이터를 분리하지 않고 전체 데이터를 학습에 사용
X_train, y_train = X, y

# LSTM 모델 생성
model = Sequential()

# LSTM 레이어 추가
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))

model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.2))

# 출력 레이어 (365일의 강수량과 기온 예측)
model.add(Dense(units=365*2))  # 365일의 강수량과 기온을 예측 (2개의 변수 예측)

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 훈련
history = model.fit(X_train, y_train, epochs=20, batch_size=32)

# 모델을 .keras 형식으로 저장
model.save('/mnt/data/lstm_weather_prediction_model.keras')

# 스케일러 저장
import joblib
joblib.dump(scaler, '/mnt/data/scaler.pkl')
