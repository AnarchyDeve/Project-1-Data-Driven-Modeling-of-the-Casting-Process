import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

# 그래프 한글 깨짐 방지
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

# 데이터 로딩
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")


# datetime & hour 생성
train['datetime'] = pd.to_datetime(train['date'] + " " + train['time'])
train['hour'] = train['datetime'].dt.hour

# shift (Day/Night 교대조) 생성

first_hour = train.loc[0, 'hour']
if 8 <= first_hour <= 19:   # 08:00 ~ 19:59 → Day
    current_shift = "Day"
else:                       # 20:00 ~ 07:59 → Night
    current_shift = "Night"

shifts = []
for i in range(len(train)):
    if i == 0:
        shifts.append(current_shift)
        continue

    if train.loc[i, 'count'] < train.loc[i-1, 'count']:  # count 감소 → 리셋
        h = train.loc[i, 'hour']
        if (8 <= h <= 19 and current_shift == "Night") or (h >= 20 or h < 8 and current_shift == "Day"):
            current_shift = "Night" if current_shift == "Day" else "Day"
    shifts.append(current_shift)

train['shift'] = shifts
print("shift 컬럼 생성 완료 (Day/Night)")

train = train.sort_values(by="datetime").reset_index(drop=True)

# ------------------------------
# 1) 글로벌 누적 count (전체 기간)
# ------------------------------
global_counts = []
accum = 0
prev_count = train.loc[0, 'count']

for i, row in train.iterrows():
    current_count = row['count']
    
    # count가 리셋된 경우
    if current_count < prev_count:
        accum += prev_count  # 지금까지 생산량을 누적
    global_counts.append(accum + current_count)
    prev_count = current_count

train['global_count'] = global_counts

# ------------------------------
# 2) 월별 누적 count
# ------------------------------
train['year_month'] = train['datetime'].dt.to_period("M")  # 연-월 단위
train['monthly_count'] = train.groupby('year_month').cumcount() + 1

# 파생 변수 생성

train['real_time'] = train['date'] + " " + train['time']
train['speed_ratio'] = train['low_section_speed'] / train['high_section_speed']
train['pressure_speed_ratio'] = train['cast_pressure'] / train['high_section_speed']

# 불필요한 컬럼 제거

drop_columns = [ 'id', 'line', 'name', 'mold_name', 'date', 'time', 'registration_time','year_month','hour','datetime']
train = train.drop(columns=drop_columns)

# real_time 컬럼을 datetime으로 변환
train['real_time'] = pd.to_datetime(train['real_time'])

train.head()
train = train.sort_values(by='real_time', ascending=True).reset_index(drop=True)


train.isna().sum()
train = train.dropna(subset=['low_section_speed'])

train[train['speed_ratio'].isna()]
train['speed_ratio'] = train['speed_ratio'].fillna(-1)



# 확인할 컬럼 목록 (passorfail, real_time, status 제외)
cols = [c for c in train.columns if c not in ['passorfail', 'real_time', 'status']]



for col in cols:
    # 기본 라인 그래프 (status 별 색상 구분)
    fig = px.line(train, x='real_time', y=col, color='status',
                  title=f"{col} 시간 분포 (불량 vs 정상)",
                  labels={'real_time': '시간', col: col, 'status': '상태'})
    
    # tryshot_signal == 'D' 인 값만 필터링
    subset = train[train['tryshot_signal'] == 'D']
    
    # 초록색 점 추가 (Scatter trace)
    fig.add_trace(
        go.Scatter(
            x=subset['real_time'],
            y=subset[col],
            mode='markers',
            marker=dict(color='green', size=6),
            name="tryshot_signal = D"
        )
    )
    
    fig.show()

train.head()

molten_temp	facility_operation_cycleTime	production_cycletime	low_section_speed	high_section_speed	molten_volume	cast_pressure 온도까지 다안나와


# 결측치는 -1로 대체
train['molten_temp'] = train['molten_temp'].fillna(-1)



# 불량/정상 라벨링
train['status'] = train['passorfail'].map({1: "불량", 0: "정상"})
# tryshot_signal == 'D' 데이터만 따로 추출
subset = train[train['tryshot_signal'] == 'D']

# 기본 라인 그래프
fig = px.line(
    train, 
    x='real_time', 
    y='molten_temp', 
    color='status',
    title="molten_temp 시간 분포 (불량 vs 정상)",
    labels={'real_time': '시간', 'molten_temp': 'molten_temp', 'status': '상태'}
)

# D인 경우 초록색 점 표시
if not subset.empty:
    fig.add_trace(
        go.Scatter(
            x=subset['real_time'],
            y=subset['molten_temp'],
            mode='markers',
            marker=dict(color='green', size=6),
            name="tryshot_signal = D"
        )
    )

fig.show()

train['mold_code'].unique() [8722, 8412, 8573, 8917, 8600]