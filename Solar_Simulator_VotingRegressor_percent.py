import pandas as pd
import numpy as np
import joblib
from tqdm.auto import tqdm
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# GridSearchCV의 진행 상황을 보여주는 Wrapper 클래스
class TqdmGridSearchCV(GridSearchCV):
    def _run_search(self, evaluate_candidates):
        """Use tqdm with GridSearchCV to display a progress bar."""
        def evaluate_candidates_with_tqdm(candidate_params):
            candidate_params = list(candidate_params)
            total_candidates = len(candidate_params)
            with tqdm(total=total_candidates, desc="Evaluating Grid", unit='param') as tqdm_out:
                results = []
                for i, parameters in enumerate(candidate_params):
                    results.append(evaluate_candidates([parameters]))
                    tqdm_out.set_description(f"Progress: {100 * (i + 1) / total_candidates:.2f}%")
                    tqdm_out.update(1)
            tqdm_out.close()
            return results
        self._evaluate_candidates = evaluate_candidates_with_tqdm
        super()._run_search(evaluate_candidates)

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
    ('창원', 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/최종 데이터/2022_최종데이터_한국남동발전_창원.csv'),
    ('구미', 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/최종 데이터/1975_2022_최종데이터_예측_구미.csv'),
    ('영흥', 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/최종 데이터/1975_2022_최종데이터_예측_영흥.csv'),
    ('예천', 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/최종 데이터/1975_2022_최종데이터_예측_예천.csv'),
    ('진주', 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/최종 데이터/1975_2022_최종데이터_예측_진주.csv'),
]

dfs = []
for location, path in tqdm(file_paths, desc="Loading data"):
    df = pd.read_csv(path)
    df['일시'] = pd.to_datetime(df['일시'])
    df = df.set_index('일시')
    df = df[['강수량(mm)', '평균기온(℃)', '최고기온(℃)', '최저기온(℃)', '총량']]
    df['총량(%)'] = df['총량'] / df['총량'].max() * 100
    dfs.append(df)

combined_df = pd.concat(dfs)
combined_df = combined_df.infer_objects()
combined_df.interpolate(inplace=True)
combined_df.dropna(inplace=True)

features = combined_df[['강수량(mm)', '평균기온(℃)', '최고기온(℃)', '최저기온(℃)']]
target = combined_df['총량(%)']

scaler_X = StandardScaler()
scaler_y = StandardScaler()

features_scaled = scaler_X.fit_transform(features)
target_scaled = scaler_y.fit_transform(target.values.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(features_scaled, target_scaled, test_size=0.2, random_state=42)

# 모델 정의 및 앙상블 생성
ensemble_model = VotingRegressor(estimators=[
    ('xgb', XGBRegressor()),
    ('rf', RandomForestRegressor()),
    ('lgbm', LGBMRegressor()),
    ('cat', CatBoostRegressor(verbose=0)),
    ('ridge', Ridge())
])

# 이전에 찾은 최적의 하이퍼파라미터 사용
param_grid = {
    'xgb__n_estimators': [200],
    'xgb__learning_rate': [0.1],
    'xgb__max_depth': [3],
    'xgb__subsample': [0.7],
    'xgb__colsample_bytree': [1],
    'xgb__gamma': [0.1],

    'rf__n_estimators': [100],
    'rf__max_depth': [15],
    'rf__min_samples_split': [2],
    'rf__min_samples_leaf': [1],
    'rf__max_features': ['sqrt'],

    'lgbm__n_estimators': [100],
    'lgbm__learning_rate': [0.1],
    'lgbm__num_leaves': [31],
    'lgbm__max_bin': [255],

    'cat__iterations': [200],
    'cat__learning_rate': [0.1],
    'cat__depth': [6],

    'ridge__alpha': [10]
}

# 하이퍼파라미터 튜닝
grid_search = TqdmGridSearchCV(ensemble_model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=3)
grid_search.fit(X_train, y_train.ravel())

# 모델 평가
y_train_pred = grid_search.best_estimator_.predict(X_train)
y_test_pred = grid_search.best_estimator_.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

# 결과 출력
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Training RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")

# 모델 저장
joblib.dump(grid_search.best_estimator_, 'best_model.pkl')
joblib.dump(scaler_X, 'scaler_X.gz')
joblib.dump(scaler_y, 'scaler_y.gz')