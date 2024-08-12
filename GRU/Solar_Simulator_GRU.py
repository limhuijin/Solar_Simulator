import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense, Input, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
import joblib

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
def load_and_combine_data(file_paths):
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
        max_capacity = data['총량'].max()
        data['상대효율'] = data['총량'] / max_capacity
        data_list.append(data)

    combined_data = pd.concat(data_list, ignore_index=True)
    return combined_data

# 데이터 전처리 함수 정의
def preprocess_data(data):
    # '일시' 열을 연, 월, 일로 변환
    data['일시'] = pd.to_datetime(data['일시'], errors='coerce')
    data['연'] = data['일시'].dt.year
    data['월'] = data['일시'].dt.month
    data['일'] = data['일시'].dt.day
    data['요일'] = data['일시'].dt.dayofweek
    data = data.drop(columns=['일시'])

    # 강수량과 1시간최다강수량의 결측값은 0으로 채움
    data.loc[:, '강수량(mm)'] = data['강수량(mm)'].fillna(0)

    # 숫자 데이터의 다른 결측값은 평균으로 채움
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data.loc[:, numeric_columns] = data.loc[:, numeric_columns].fillna(data[numeric_columns].mean())
    
    return data

# 데이터 로드 및 전처리
data = load_and_combine_data(file_paths)
data = preprocess_data(data)

# 특성과 목표 변수 설정
features = ['강수량(mm)', '평균기온(℃)', '최고기온(℃)', '최저기온(℃)', '평균풍속(m/s)', '평균습도(%rh)', '월', '일']
X = data[features]
y = data['총량']

# 데이터 정규화
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

# GRU 입력 형식에 맞게 데이터 변환
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=42)
val_losses = []

# 조기 중단 및 학습률 감소 콜백 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.0001)

for train_index, val_index in kf.split(X_scaled):
    X_train, X_val = X_scaled[train_index], X_scaled[val_index]
    y_train, y_val = y_scaled[train_index], y_scaled[val_index]
    
    # GRU 모델 생성
    model = Sequential()
    model.add(Input(shape=(1, X_train.shape[2])))
    model.add(GRU(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(64))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    # 모델 컴파일
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # 모델 학습
    history = model.fit(X_train, y_train, epochs=1000, batch_size=64, validation_data=(X_val, y_val),
                        callbacks=[early_stopping, reduce_lr])
    
    # 모델 평가
    val_loss = model.evaluate(X_val, y_val)
    val_losses.append(val_loss)
    print(f'Validation Loss: {val_loss}')

# 평균 검증 손실 출력
print(f'Average Validation Loss: {np.mean(val_losses)}')

# 모델 저장
model.save('C:/Users/user/Desktop/coding/Solar_Simulator/model/model.keras')

# 스케일러 저장
joblib.dump(scaler_X, 'C:/Users/user/Desktop/coding/Solar_Simulator/model/scaler_X.gz')
joblib.dump(scaler_y, 'C:/Users/user/Desktop/coding/Solar_Simulator/model/scaler_y.gz')
