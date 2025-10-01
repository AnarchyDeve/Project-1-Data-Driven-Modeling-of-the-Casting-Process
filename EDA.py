################ 몰드 템프 전처리

import pandas as pd
import plotly.express as px

train = pd.read_csv('./data/train.csv')

train['molten_temp'].value_counts()

# registration_time을 datetime 형식으로 변환
train['registration_time'] = pd.to_datetime(train['registration_time'], errors='coerce')

# 시계열 그래프 (molten_temp 변화)
fig = px.line(
    train,
    x="registration_time",
    y="molten_temp",
    title="Molten Temperature over Time",
    labels={"registration_time": "등록 일시", "molten_temp": "용탕 온도 (℃)"},
    template="plotly_dark"
)

# 줌인/아웃 및 hover 기능 활성화
fig.update_xaxes(rangeslider_visible=True)
fig.show()
# molten_temp별 빈도 계산 후 작은 값부터 정렬
molten_counts = train['molten_temp'].value_counts().sort_index()

# 작은 값부터 50개만 확인
molten_counts.head(50)


# 보고 싶은 값 리스트
target_vals = [0.0, 7.0, 70.0, 71.0, 72.0, 73.0]

# 각 값별 passorfail 비율 계산
ratio_df = (
    train[train['molten_temp'].isin(target_vals)]
    .groupby('molten_temp')['passorfail']
    .value_counts(normalize=True)  # 비율 계산
    .unstack(fill_value=0)         # 열로 펼치기
    .rename(columns={0.0: "양품비율", 1.0: "불량비율"})
)

print(ratio_df)


# 0도, 7도 조건
mask = train['molten_temp'].isin([70.0, 71.0,72,73])
idx_list = train[mask].index

# 전후 3행씩 확인
for i in idx_list:
    print(train.loc[i-3:i+3, ["registration_time", "molten_temp", "passorfail"]])
    print("="*60)


# 7도인 행 인덱스 추출
idx_list = train[train['molten_temp'] == 7.0].index

# 전후 3행씩 확인
for i in idx_list:
    print(f"🔹 molten_temp=7.0 at index {i}")
    print(train.loc[i-3:i+3, ["registration_time", "molten_temp", "passorfail"]])
    print("="*60)



import pandas as pd
import numpy as np
import plotly.express as px

# ----- 데이터 로드
train = pd.read_csv("./data/train.csv")

# ----- datetime 변환 (시계열 그래프용)
train['registration_time'] = pd.to_datetime(train['registration_time'], errors='coerce')

# ----- Step 1. 100 이하 → NaN 변환
train['molten_temp_clean'] = train['molten_temp'].where(train['molten_temp'] >= 100, np.nan)

# ----- Step 2. NaN을 앞뒤 평균으로 보정
def fill_nan_with_neighbors(series):
    vals = series.copy()
    for i in range(len(vals)):
        if pd.isna(vals.iloc[i]):
            prev_idx = i - 1
            next_idx = i + 1

            # 앞 탐색
            while prev_idx >= 0 and pd.isna(vals.iloc[prev_idx]):
                prev_idx -= 1
            # 뒤 탐색
            while next_idx < len(vals) and pd.isna(vals.iloc[next_idx]):
                next_idx += 1

            # 앞뒤 모두 값이 있으면 평균
            if prev_idx >= 0 and next_idx < len(vals):
                vals.iloc[i] = np.mean([vals.iloc[prev_idx], vals.iloc[next_idx]])
            elif prev_idx >= 0:  # 앞만 있으면 앞값
                vals.iloc[i] = vals.iloc[prev_idx]
            elif next_idx < len(vals):  # 뒤만 있으면 뒤값
                vals.iloc[i] = vals.iloc[next_idx]
            # 둘 다 없으면 그대로 NaN 유지
    return vals

train['molten_temp_filled'] = fill_nan_with_neighbors(train['molten_temp_clean'])

# ----- Plotly 시각화 (보정 전 vs 보정 후 비교)
fig = px.line(
    train,
    x="registration_time",
    y=["molten_temp", "molten_temp_filled"],
    labels={"value": "Molten Temperature (℃)", "registration_time": "등록 일시"},
    title="Molten Temperature (Before vs After Correction)"
)

fig.update_xaxes(rangeslider_visible=True)
fig.show()
train['molten_temp_filled'].value_counts()

train['molten_temp'] = train['molten_temp_filled']
train['molten_temp'].value_counts()

import pandas as pd
import plotly.express as px

train = pd.read_csv("./data/train.csv")

# datetime 변환
train['registration_time'] = pd.to_datetime(train['registration_time'], errors='coerce')

# 시계열 Plotly 그래프
fig = px.line(
    train,
    x="registration_time",
    y="molten_volume",  # ✅ 정확한 컬럼명 사용
    labels={
        "registration_time": "등록 일시",
        "molten_volume": "설비 작동 사이클 시간"
    },
    title="Facility Operation CycleTime over Time"
)

fig.update_xaxes(rangeslider_visible=True)
fig.show()

train[train['facility_operation_cycleTime'] >= 300]

train.columns

train.isna().sum()

import numpy as np
import pandas as pd
from itertools import groupby
from operator import itemgetter

# 데이터 로딩
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

def extract_cycles(series, threshold=0.5):
    """스무딩된 시계열에서 전환점 기반 사이클 추출 (임계치 적용)"""
    series = np.asarray(series)
    diff = np.diff(series)

    turns = []
    for i in range(1, len(diff)):
        if np.sign(diff[i]) != np.sign(diff[i-1]) and abs(diff[i]) > threshold:
            turns.append(i)
    turns = np.array(turns)

    cycles = []
    for i in range(len(turns)-1):
        cycles.append(series[turns[i]:turns[i+1]])
    return cycles



def fill_with_cycles(series, window=5, n_cycles=5):
    values = series.values.copy()
    nan_idx = np.where(np.isnan(values))[0]

    # 스무딩 적용
    smooth = pd.Series(values).rolling(window=window, center=True, min_periods=1).mean().values

    # 연속 NaN 블록 찾기
    nan_groups = []
    for k, g in groupby(enumerate(nan_idx), lambda ix: ix[0]-ix[1]):
        block = list(map(itemgetter(1), g))
        nan_groups.append(block)

    for block in nan_groups:
        start, end = block[0], block[-1]
        block_len = len(block)

        prev_vals = smooth[:start][~np.isnan(smooth[:start])]
        next_vals = smooth[end+1:][~np.isnan(smooth[end+1:])]

        prev_cycles = extract_cycles(prev_vals) if len(prev_vals) > 10 else []
        next_cycles = extract_cycles(next_vals) if len(next_vals) > 10 else []

        if len(prev_cycles) >= n_cycles:
            pattern = np.concatenate(prev_cycles[-n_cycles:])
        elif len(next_cycles) >= n_cycles:
            pattern = np.concatenate(next_cycles[:n_cycles])
        elif len(prev_cycles) > 0:
            pattern = np.concatenate(prev_cycles)
        elif len(next_cycles) > 0:
            pattern = np.concatenate(next_cycles)
        else:
            # fallback: 앞뒤 값으로 선형보간
            pattern = np.linspace(values[start-1], values[end+1], block_len)

        reps = int(np.ceil(block_len / len(pattern))) + 1
        filled = np.tile(pattern, reps)[:block_len]
        values[start:end+1] = filled

    return pd.Series(values, index=series.index)

# 1) 이상치 → NaN 처리
mask = train["molten_volume"] > 200
outlier_idx = train.index[mask]
outlier_val = train.loc[mask, "molten_volume"]
train.loc[mask, "molten_volume"] = np.nan

# 2) 패턴 기반 보간 실행
train["molten_volume_interp"] = fill_with_cycles(train["molten_volume"], window=5, n_cycles=5)

# 3) 원본 복원 (이상치 되살리기)
train.loc[outlier_idx, "molten_volume"] = outlier_val

cast_pressure                       1
biscuit_thickness                   1
-
lower_mold_temp1                    1
lower_mold_temp2                    1
lower_mold_temp3                  313
sleeve_temperature                  1
physical_strength                   1
Coolant_temperature                 1



import pandas as pd

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

train.columns
train

import pandas as pd

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# 방법 1: groupby + size
date_counts = train.groupby('time').size().reset_index(name='count')
print(date_counts)

# 방법 2: value_counts (더 간단)
date_counts2 = train['time'].value_counts().sort_index()
print(date_counts2)

train["time"] = pd.to_datetime(train["time"], errors="coerce")  # 날짜로 변환
train["day"] = train["time"].dt.date   # 날짜만 추출 (예: 2019-01-02)


# 하루 총 생산량
daily_total = train.groupby("day")["count"].max().reset_index(name="daily_total")


# 하루-몰드코드별 생산량
daily_mold = train.groupby(["day", "mold_code"])["count"].max().reset_index(name="mold_count")

# 하루 총합과 합치기
daily_mold = daily_mold.merge(daily_total, on="day")

# 비율 계산
daily_mold["ratio"] = daily_mold["mold_count"] / daily_mold["daily_total"]


import matplotlib.pyplot as plt

pivot_mold = daily_mold.pivot(index="day", columns="mold_code", values="ratio").fillna(0)
pivot_mold.plot(kind="bar", stacked=True, figsize=(14,6))
plt.title("날짜별 몰드코드 비율")
plt.ylabel("비율")
plt.show()

import seaborn as sns

sns.scatterplot(data=daily_mold, x="ratio", y="daily_total", hue="mold_code")
plt.title("몰드코드 비율 vs 하루 생산량")
plt.show()


import pandas as pd

# 데이터 로드
train = pd.read_csv("./data/train.csv")
train["time"] = pd.to_datetime(train["time"], errors="coerce")
train["day"] = train["time"].dt.date

# ---- 1. 하루 실제 산출량 (count 기준) ----
daily_actual = train.groupby("day")["count"].agg(["min", "max"]).reset_index()
daily_actual["daily_actual"] = daily_actual["max"] - daily_actual["min"] + 1

# ---- 2. 몰드코드별 평균 facility cycle time ----
cycle_stats = (
    train.groupby(["day", "mold_code"])["facility_operation_cycleTime"]
    .mean()
    .reset_index(name="avg_facility_cycleTime")
)

# ---- 3. 이론적 산출량 (24시간 기준) ----
WORK_TIME = 24 * 60 * 60  # 하루 24시간 = 86,400초
cycle_stats["theoretical_output"] = (WORK_TIME / cycle_stats["avg_facility_cycleTime"]).round()

# ---- 4. merge (day 기준) ----
result = cycle_stats.merge(daily_actual[["day", "daily_actual"]], on="day")

# ---- 5. 오차율 계산 ----
result["error_rate(%)"] = (
    (result["daily_actual"] - result["theoretical_output"]) 
    / result["theoretical_output"] * 100
).round(2)

# ---- 6. 결과 확인 ----
print(result.head(15))
