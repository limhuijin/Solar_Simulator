import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 각 지역의 최대 발전 가능량 (예시 값, 실제 값으로 변경 필요)
max_capacity = {
    '구미': 992,
    '영흥': 1000,
    '진주': 905,
    '창원': 77
}

# 데이터 파일 경로 설정
file_paths = [
    ('영흥', 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/최종 데이터/2018_최종데이터_한국남동발전_영흥.csv'),
    ('영흥', 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/최종 데이터/2019_최종데이터_한국남동발전_영흥.csv'),
    ('진주', 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/최종 데이터/2019_최종데이터_한국남동발전_진주.csv'),
    ('창원', 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/최종 데이터/2019_최종데이터_한국남동발전_창원.csv'),
    ('구미', 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/최종 데이터/2020_최종데이터_한국남동발전_구미.csv'),
    ('영흥', 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/최종 데이터/2020_최종데이터_한국남동발전_영흥.csv'),
    ('진주', 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/최종 데이터/2020_최종데이터_한국남동발전_진주.csv'),
    ('창원', 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/최종 데이터/2020_최종데이터_한국남동발전_창원.csv'),
    ('영흥', 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/최종 데이터/2021_최종데이터_한국남동발전_영흥.csv'),
    ('진주', 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/최종 데이터/2021_최종데이터_한국남동발전_진주.csv'),
    ('구미', 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/최종 데이터/2022_최종데이터_한국남동발전_구미.csv'),
    ('영흥', 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/최종 데이터/2022_최종데이터_한국남동발전_영흥.csv'),
    ('진주', 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/최종 데이터/2022_최종데이터_한국남동발전_진주.csv'),
    ('창원', 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/최종 데이터/2022_최종데이터_한국남동발전_창원.csv')
]

# 데이터 로드 및 결합 함수 정의
def load_and_combine_data(file_paths, max_capacity):
    data_list = []
    for region, file_path in file_paths:
        try:
            data = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                data = pd.read_csv(file_path, encoding='latin1')
            except UnicodeDecodeError:
                data = pd.read_csv(file_path, encoding='cp949')
        
        data.columns = data.columns.str.strip()
        if '총량' not in data.columns:
            print(f"Warning: '총량' column not found in {file_path}")
            continue

        data['지역'] = region
        data['상대효율'] = data['총량'] / max_capacity[region]
        data_list.append(data)

    combined_data = pd.concat(data_list, ignore_index=True)
    return combined_data

# 데이터 로드 및 결합
data = load_and_combine_data(file_paths, max_capacity)

# '일시' 열을 연, 월, 일로 변환
data['일시'] = pd.to_datetime(data['일시'], errors='coerce')
data['연'] = data['일시'].dt.year
data['월'] = data['일시'].dt.month
data['일'] = data['일시'].dt.day
data = data.drop(columns=['일시'])

# 결측값 처리 및 이상치 제거
data.fillna(0, inplace=True)

# 특성과 목표 변수 설정
features = ['강수량(mm)', '1시간최다강수량(mm)', '평균기온(℃)', '최고기온(℃)', '최저기온(℃)', 
            '평균풍속(m/s)', '최대풍속(m/s)', '최대풍속풍향(deg)', '최대순간풍속(m/s)', 
            '최대순간풍속풍향(deg)', '평균습도(%rh)', '최저습도(%rh)']
X = data[features]
y = data['총량']

# 데이터 정규화
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# LSTM 입력 형식에 맞게 데이터 변환
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# LSTM 모델 생성
model = Sequential()
model.add(LSTM(128, input_shape=(1, X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64))
model.add(Dropout(0.2))
model.add(Dense(1))

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 학습
history = model.fit(X_train, y_train, epochs=500, batch_size=32, validation_split=0.2)

# 모델 평가
loss = model.evaluate(X_test, y_test)
print(f'Model Loss: {loss}')

# 모델 저장
model.save('C:/Users/user/Desktop/coding/Solar_Simulator/model/model.keras')
