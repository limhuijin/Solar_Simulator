import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 데이터 파일 경로
file_path = 'C:/Users/user/Desktop/coding/Solar_Simulator/csv/태양광 데이터/2021_태양광데이터_한국남동발전_예천.csv'

# 데이터 로드
data = pd.read_csv(file_path, encoding='utf-8')

# '년월일' 열을 날짜 형식으로 변환
data['년월일'] = pd.to_datetime(data['년월일'], errors='coerce')

# 최대 값 계산
max_total = data['총량'].max()

# 효율 계산 (현재 값 / 최대 값 * 100)
data['효율'] = (data['총량'] / max_total) * 100

# 날짜별로 그룹화하여 평균 효율 계산
data['날짜'] = data['년월일'].dt.date
daily_efficiency = data.groupby('날짜')['효율'].mean()

# 월별로 그룹화하여 평균 효율 계산
data['월'] = data['년월일'].dt.month
monthly_efficiency = data.groupby('월')['효율'].mean()

# 24기간으로 나누어 그룹화하여 평균 효율 계산 (각 기간은 약 15일)
data['기간'] = (data['년월일'].dt.dayofyear - 1) // 15
biweekly_efficiency = data.groupby('기간')['효율'].mean()

# 시각화 함수 정의
def plot_efficiency(x, y, xlabel, ylabel, title, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, label='Efficiency (%)', color='b')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# 365일 기준 효율 시각화
plot_efficiency(daily_efficiency.index, daily_efficiency, 'Day of the Year', 'Efficiency (%)', 
                'Daily Solar Energy Efficiency', '365.png')

# 12개월 기준 효율 시각화
plot_efficiency(monthly_efficiency.index, monthly_efficiency, 'Month', 'Efficiency (%)', 
                'Monthly Solar Energy Efficiency', '12.png')

# 24기간 기준 효율 시각화
plot_efficiency(biweekly_efficiency.index, biweekly_efficiency, 'Biweekly Period', 'Efficiency (%)', 
                'Biweekly Solar Energy Efficiency', '24.png')

# 저장된 이미지 파일 경로 출력
print("Images saved:")
print(" - daily_efficiency.png")
print(" - monthly_efficiency.png")
print(" - biweekly_efficiency.png")
