import pandas as pd
import joblib
from sklearn.ensemble import VotingRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np

# 데이터 로드 및 전처리
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

dfs = []
for location, path in file_paths:
    df = pd.read_csv(path)
    df['일시'] = pd.to_datetime(df['일시'])
    df = df.set_index('일시')
    df = df[['강수량(mm)', '평균기온(℃)', '최고기온(℃)', '최저기온(℃)', '평균풍속(m/s)', '총량']]
    df['총량(%)'] = df['총량'] / df['총량'].max() * 100  # 각 파일별로 최대값 대비 퍼센트로 변환
    dfs.append(df)

combined_df = pd.concat(dfs)
combined_df = combined_df.infer_objects()
combined_df.interpolate(inplace=True)
combined_df.dropna(inplace=True)

features = combined_df[['강수량(mm)', '평균기온(℃)', '최고기온(℃)', '최저기온(℃)', '평균풍속(m/s)']]
target = combined_df['총량(%)']

scaler_X = StandardScaler()
scaler_y = StandardScaler()

features_scaled = scaler_X.fit_transform(features)
target_scaled = scaler_y.fit_transform(target.values.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(features_scaled, target_scaled, test_size=0.2, random_state=42)

# 모델 정의
xgb_model = XGBRegressor()
rf_model = RandomForestRegressor()
lgbm_model = LGBMRegressor()
cat_model = CatBoostRegressor(verbose=0)
ridge_model = Ridge()

# 앙상블 모델 정의
ensemble_model = VotingRegressor(estimators=[
    ('xgb', xgb_model),
    ('rf', rf_model),
    ('lgbm', lgbm_model),
    ('cat', cat_model),
    ('ridge', ridge_model)
])

# 하이퍼파라미터 튜닝
param_grid = {
    'xgb__n_estimators': [100, 200],
    'xgb__learning_rate': [0.01, 0.1],
    'xgb__max_depth': [3, 5],
    'xgb__reg_alpha': [0, 0.1],  # L1 regularization
    'xgb__reg_lambda': [1, 10],  # L2 regularization
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [10, 15],
    'rf__max_features': ['sqrt', 'log2'],  # feature selection
    'lgbm__n_estimators': [100, 200],
    'lgbm__learning_rate': [0.01, 0.1],
    'lgbm__num_leaves': [31, 50],
    'cat__iterations': [100, 200],
    'cat__learning_rate': [0.01, 0.1],
    'cat__depth': [6, 10],
    'ridge__alpha': [1, 10, 100]  # Regularization strength
}

grid_search = GridSearchCV(estimator=ensemble_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train.ravel())

best_model = grid_search.best_estimator_

# 모델 평가
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"Best parameters found: {grid_search.best_params_}")
print(f"Training RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")

# 모델 저장
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(scaler_X, 'scaler_X.gz')
joblib.dump(scaler_y, 'scaler_y.gz')
