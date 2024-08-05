import pandas as pd
import numpy as np
from prophet import Prophet
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
    for location, file_path in file_paths:
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

        max_capacity = data['총량'].max()
        data['상대효율'] = data['총량'] / max_capacity
        data['지역'] = location
        data_list.append(data)

    combined_data = pd.concat(data_list, ignore_index=True)
    return combined_data

# 데이터 로드 및 전처리
data = load_and_combine_data(file_paths)

# '일시' 열을 연, 월, 일로 변환
data['일시'] = pd.to_datetime(data['일시'], errors='coerce')

# NaN 값 제거
data = data.dropna(subset=['일시'])

data['연'] = data['일시'].dt.year
data['월'] = data['일시'].dt.month
data['일'] = data['일시'].dt.day
data['요일'] = data['일시'].dt.dayofweek

# 결측값 처리 (중위수 사용)
data['강수량(mm)'] = data['강수량(mm)'].fillna(0)
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

# 추가적인 특성 생성 (로그 변환 및 스케일링)
data['log_총량'] = np.log1p(data['총량'])
data['강수량(mm)_scaled'] = (data['강수량(mm)'] - data['강수량(mm)'].mean()) / data['강수량(mm)'].std()
data['평균기온(℃)_scaled'] = (data['평균기온(℃)'] - data['평균기온(℃)'].mean()) / data['평균기온(℃)'].std()

# Prophet 모델을 위한 데이터 준비
df = data[['일시', 'log_총량', '강수량(mm)_scaled', '평균기온(℃)_scaled']].rename(columns={'일시': 'ds', 'log_총량': 'y'})

# Prophet 모델 생성 및 하이퍼파라미터 튜닝
model = Prophet(
    changepoint_prior_scale=0.1,  # 트렌드 변화 / 기본값은 0.05
    seasonality_prior_scale=10.0,  # 계절성 변동 / 기본값은 10.0
    holidays_prior_scale=10.0,  # 공휴일 효과 / 기본값은 10.0
    seasonality_mode='additive',  # 계절성 모드 / 기본값은 'additive'
    n_changepoints=50  # 트렌드 변화점의 수 / 기본값은 25
)

# 중요한 리그레서 추가 (prior_scale 조정하여 중요도 증가)
model.add_regressor('강수량(mm)_scaled', prior_scale=20.0)
model.add_regressor('평균기온(℃)_scaled', prior_scale=20.0)

# 모델 학습
model.fit(df)

# 모델 저장
model_path = 'C:/Users/user/Desktop/coding/Solar_Simulator/model/model_prophet.pkl'
joblib.dump(model, model_path)

print(f'Model saved to {model_path}')
