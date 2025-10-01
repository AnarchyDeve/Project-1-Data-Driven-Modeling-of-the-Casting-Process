##############################################################
# EDA with Smoothing + RangeSlider for ALL numeric features
##############################################################
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ----- 데이터 로드
train = pd.read_csv("./data/train.csv")

# ----- datetime 생성
train["datetime"] = pd.to_datetime(
    train["date"].astype(str) + " " + train["time"].astype(str),
    errors="coerce",
    infer_datetime_format=True
)

# ----- 수치형 컬럼 추출
num_cols = train.select_dtypes(include=[np.number]).columns.tolist()

# ----- 스무딩 함수 (데이터 개수 유지)
def smooth_series(series, window=20):
    return series.rolling(window=window, center=True, min_periods=1).mean()

# ----- 변수별 플롯
for col in num_cols:
    if col == "passorfail":  # 타깃은 제외
        continue
    
    fig = go.Figure()
    for mold, df_sub in train.groupby("mold_code"):
        smoothed = smooth_series(df_sub[col], window=20)
        fig.add_trace(go.Scatter(
            x=df_sub["datetime"],
            y=smoothed,
            mode="lines",
            name=f"Mold {mold}"
        ))

    # RangeSlider & RangeSelector 추가
    fig.update_layout(
        title=f"{col} (Smoothed, with RangeSlider)",
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),  # 아래 작은 그래프
            type="date"
        ),
        hovermode="x unified"
    )
    
    fig.show()
