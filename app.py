import pandas as pd
import joblib
import matplotlib.pyplot as plt
from shiny import App, ui, render, reactive
from shiny.ui import update_slider, update_numeric, update_select, update_navs
import seaborn as sns
import pathlib
import plotly.express as px
from shinywidgets import render_plotly, output_widget
import numpy as np
import matplotlib
from sklearn.metrics import pairwise_distances
import os
from matplotlib import font_manager
import matplotlib.pyplot as plt
import plotly.io as pio
import calendar
import datetime
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy import stats


matplotlib.use("Agg")   # Tkinter 대신 Agg backend 사용 (GUI 필요 없음)

app_dir = pathlib.Path(__file__).parent
# ===== 한글 깨짐 방지 설정 =====
plt.rcParams["font.family"] = "Malgun Gothic"   # 윈도우: 맑은 고딕
plt.rcParams["axes.unicode_minus"] = False      # 마이너스 기호 깨짐 방지

# 폰트 파일 경로
APP_DIR = os.path.dirname(os.path.abspath(__file__))
font_path = os.path.join(APP_DIR, "www", "fonts", "NanumGothic-Regular.ttf")

# 폰트 적용
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "NanumGothic"  # Matplotlib
    print(f"✅ 한글 폰트 적용됨: {font_path}")
else:
    plt.rcParams["font.family"] = "sans-serif"
    print(f"⚠️ 한글 폰트 파일 없음 → {font_path}")

# 마이너스 부호 깨짐 방지
plt.rcParams["axes.unicode_minus"] = False

# Plotly 기본 폰트 설정
pio.templates["nanum"] = pio.templates["plotly_white"].update(
    layout_font=dict(family="NanumGothic")
)
pio.templates.default = "nanum"

# ===== 모델 불러오기 =====
MODEL_PATH = "./models/model_2.pkl"
model = joblib.load(MODEL_PATH)

# ===== 데이터 불러오기 =====
df_raw = pd.read_csv("./data/train_raw.csv")

# ★ 특정 이상치 행 제거
df_raw = df_raw[
    (df_raw["low_section_speed"] != 65535) &
    (df_raw["lower_mold_temp3"] != 65503) &
    (df_raw["physical_strength"] != 65535)
]

# 예측용 데이터도 동일 처리
df_predict = pd.read_csv("./data/train.csv")
df_predict["pressure_speed_ratio"] = df_predict["pressure_speed_ratio"].replace([np.inf, -np.inf], np.nan)


# 예측 탭용 (모델 input 그대로)
df_predict = pd.read_csv("./data/train.csv")
df_predict["pressure_speed_ratio"] = df_predict["pressure_speed_ratio"].replace([np.inf, -np.inf], np.nan)

df_predict = df_predict[
    (df_predict["low_section_speed"] != 65535) &
    (df_predict["lower_mold_temp3"] != 65503) &
    (df_predict["physical_strength"] != 65535)
]

# 탐색 탭용 (필터링/EDA)
drop_cols_explore = ["id","line","name","mold_name","date","time", "registration_time"]
df_explore = df_raw.drop(columns=drop_cols_explore, errors="ignore")  # ← 안전하게
# mold_code는 남김


# 전처리 후 데이터 (모델 학습용)
df_processed = pd.read_csv("./data/processed_train.csv")

# 컬럼 이름 표준화
df_processed.columns = df_processed.columns.str.strip().str.lower()
# 원본 탐색 데이터도 동일하게
df_explore.columns = df_explore.columns.str.strip().str.lower()

# 혹시 passorfail이 인덱스로 들어갔다면 컬럼으로 리셋
if "passorfail" not in df_processed.columns and "passorfail" in df_processed.index.names:
    df_processed = df_processed.reset_index()


# ✅ 파생 변수 자동 추가
derived_cols = ["speed_ratio", "pressure_speed_ratio"]
for col in derived_cols:
    if col in df_predict.columns:
        df_explore[col] = df_predict[col]

# 예측에서 제외할 컬럼
drop_cols = [
    "real_time",   # registration_time → real_time
    "passorfail",
    # "count",
    # "global_count",
    # "monthly_count",
    # "speed_ratio",
	# "pressure_speed_ratio",
    # "shift",
]
used_columns = df_predict.drop(columns=drop_cols).columns

# 그룹 분류
cat_cols = ["mold_code","working","emergency_stop","heating_furnace", "shift", "tryshot_signal"]
num_cols = [c for c in used_columns if c not in cat_cols]

# ===== 라벨 맵 =====
label_map = {
    # 기본 정보 관련
    "id": "고유 번호",
    "line": "생산 라인 이름",
    "name": "장비 이름",
    "mold_name": "금형 이름",
    "time": "측정 날짜",
    "date": "측정 시간",

    # 공정 상태 관련
    "count": "누적 제품 개수",
    "working": "장비 가동 여부 (가동 / 멈춤 등)",
    "emergency_stop": "비상 정지 여부 (ON / OFF)",
    "registration_time": "데이터 등록 시간",
    "tryshot_signal": "측정 딜레이 여부",

    # 용융 단계
    "molten_temp": "용융 온도",
    "heating_furnace": "용해로 정보",

    # 충진 단계
    "sleeve_temperature": "주입 관 온도",
    "ems_operation_time": "전자 교반(EMS) 가동 시간",
    "EMS_operation_time": "전자 교반(EMS) 가동 시간",
    "low_section_speed": "하위 구간 주입 속도",
    "high_section_speed": "상위 구간 주입 속도",
    "mold_code": "금형 코드",
    "molten_volume": "주입한 금속 양",
    "cast_pressure": "주입 압력",

    # 냉각 단계
    "upper_mold_temp1": "상부1 금형 온도",
    "upper_mold_temp2": "상부2 금형 온도",
    "upper_mold_temp3": "상부3 금형 온도",
    "lower_mold_temp1": "하부1 금형 온도",
    "lower_mold_temp2": "하부2 금형 온도",
    "lower_mold_temp3": "하부3 금형 온도",
    "coolant_temperature": "냉각수 온도",
    "Coolant_temperature": "냉각수 온도",

    # 공정 속도 관련
    "facility_operation_cycletime": "장비 전체 사이클 시간",
    "facility_operation_cycleTime": "장비 전체 사이클 시간",
    "production_cycletime": "실제 생산 사이클 시간",

    # 품질 및 성능
    "biscuit_thickness": "주조물 두께",
    "physical_strength": "제품 강도",

    # 평가
    "passorfail": "합격/불합격",

    "global_count": "전체 누적 개수",
    "monthly_count": "월간 누적 개수",
    "speed_ratio": "상/하부 주입 속도 비율",
	"pressure_speed_ratio": "주입 압력 비율",
    "shift": "주/야간 교대",
}


# ===== 라벨 정의 (표시 텍스트 = 한글, 실제 var = 변수명) =====
labels = [
    {"id": "label1", "text": label_map["upper_mold_temp1"], "var": "upper_mold_temp1",
     "x": 200, "y": 85, "w": 120, "h": 30,
     "arrow_from": (260, 115), "arrow_to": (400, 195)}, 

    {"id": "label2", "text": label_map["lower_mold_temp1"], "var": "lower_mold_temp1",
     "x": 650, "y": 85, "w": 120, "h": 30,
     "arrow_from": (710, 115), "arrow_to": (580, 195)}, 

    {"id": "label3", "text": label_map["cast_pressure"], "var": "cast_pressure",
     "x": 900, "y": 285, "w": 100, "h": 30,
     "arrow_from": (950, 315), "arrow_to": (780, 395)}, 

    {"id": "label4", "text": label_map["molten_volume"], "var": "molten_volume",
     "x": 700, "y": 185, "w": 120, "h": 30,
     "arrow_from": (760, 215), "arrow_to": (780, 315)}, 

    {"id": "label5", "text": label_map["sleeve_temperature"], "var": "sleeve_temperature",
     "x": 670, "y": 435, "w": 120, "h": 30,
     "arrow_from": (730, 435), "arrow_to": (600, 395)},  

    {"id": "label6", "text": label_map["high_section_speed"], "var": "high_section_speed",
     "x": 400, "y": 105, "w": 160, "h": 30,
     "arrow_from": (480, 135), "arrow_to": (510, 215)}, 

    {"id": "label7", "text": label_map["low_section_speed"], "var": "low_section_speed",
     "x": 400, "y": 455, "w": 160, "h": 30,
     "arrow_from": (480, 455), "arrow_to": (510, 355)},
]

def get_label(col): return label_map.get(col, col)

# ===== Helper: 슬라이더 + 인풋 =====
def make_num_slider(col):
    return ui.div(
        ui.input_slider(
            f"{col}_slider", get_label(col),
            min=int(df_predict[col].min()), max=int(df_predict[col].max()),
            value=int(df_predict[col].mean()), width="100%"
        ),
        ui.input_numeric(col, "", value=int(df_predict[col].mean()), width="110px"),
        style="display: flex; align-items: center; gap: 8px; justify-content: space-between;"
    )

# ===== 범주형 없음도 추가 ========
def make_select(col, label=None, width="100%"):
    label = label if label else get_label(col)
    if(col == "tryshot_signal"):
        choices = ["없음"] + sorted(df_predict[col].dropna().unique().astype(str))
    else:
        choices = sorted(df_predict[col].dropna().unique().astype(str)) + ["없음"]
    return ui.input_select(col, label, choices=choices, width=width)

def make_svg(labels):
    parts = []
    for lbl in labels:
        # 화살표 시작점: arrow_from 있으면 사용, 없으면 중앙
        if "arrow_from" in lbl:
            cx, cy = lbl["arrow_from"]
        else:
            cx = lbl["x"] + lbl["w"]/2
            cy = lbl["y"] + lbl["h"]/2

        x2, y2 = lbl["arrow_to"]
        text = label_map.get(lbl["var"], lbl["var"])

        parts.append(f"""
        <g>
        <rect x="{lbl['x']}" y="{lbl['y']}" width="{lbl['w']}" height="{lbl['h']}" 
                fill="#e0e6ef" stroke="black"/>
        <text x="{lbl['x'] + lbl['w']/2}" y="{lbl['y'] + lbl['h']/2}" 
                fill="black" font-size="14" font-weight="bold"
                text-anchor="middle" dominant-baseline="middle">{text}</text>
        <line x1="{cx}" y1="{cy}" x2="{x2}" y2="{y2}" 
                stroke="red" marker-end="url(#arrow)"/>
        </g>
        """)
    return "\n".join(parts)

svg_code = f"""
<svg width="1000" height="500" xmlns="http://www.w3.org/2000/svg"
     style="background:url('die-castings.gif'); background-size:cover;">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L6,3 z" fill="red"/>
    </marker>
  </defs>
  {make_svg(labels)}
</svg>
"""

# ===== CSS (카드 전체 클릭영역) =====
card_click_css = """
<style>
/* 개요 전용 카드만 hover 효과 */
.overview-card {
    transition: transform 0.2s ease, box-shadow 0.2s ease, background-color 0.2s ease;
    position: relative;
}

.overview-card:hover {
    background-color: #f8f9fa !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    transform: translateY(-2px);
}

/* 카드 전체를 클릭 가능하게 하는 투명 버튼 */
.card-link {
    position: absolute;
    inset: 0;
    z-index: 10;
    cursor: pointer;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
.card-link:hover,
.card-link:focus,
.card-link:active {
    background: transparent !important;
    box-shadow: none !important;
}
</style>
"""

# ========== 데이터 준비 ==========
train = pd.read_csv("./data/train_raw.csv")
train["time"] = pd.to_datetime(train["time"], errors="coerce")
train["day"] = train["time"].dt.date
# 몰드코드별 요약
mold_cycle = (
    train.groupby("mold_code")["facility_operation_cycleTime"]
    .mean()
    .reset_index(name="avg_facility_cycleTime")
)
mold_cycle["daily_capacity"] = (86400 / mold_cycle["avg_facility_cycleTime"]).round()

daily_actual = train.groupby(["day", "mold_code"])["count"].agg(["min", "max"]).reset_index()
daily_actual["daily_actual"] = daily_actual["max"] - daily_actual["min"] + 1

mold_stats = daily_actual.groupby("mold_code")["daily_actual"].agg(
    min_prod="min", max_prod="max", avg_prod="mean"
).reset_index()

mold_summary = pd.merge(mold_cycle, mold_stats, on="mold_code")

# mold_code를 문자열로 변환
mold_summary["mold_code"] = mold_summary["mold_code"].astype(int).astype(str)
codes = list(mold_summary["mold_code"])
last_code = codes[-1]

# 색상 팔레트
cmap = cm.get_cmap("tab10", len(codes))
mold_colors = {code: mcolors.to_hex(cmap(i)) for i, code in enumerate(codes)}

# ================================
# 권장 세팅값 계산
# ================================
def smooth_series(series, window=5):
    smoothed = series.rolling(window=window, center=True, min_periods=1).mean()
    Q1, Q3 = smoothed.quantile(0.25), smoothed.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    filtered = smoothed[(smoothed >= lower) & (smoothed <= upper)]
    return filtered.dropna()

setting_cols = [
    "molten_temp",
    "upper_mold_temp1","upper_mold_temp2","upper_mold_temp3",
    "lower_mold_temp1","lower_mold_temp2","lower_mold_temp3",
    "sleeve_temperature","cast_pressure","biscuit_thickness",
    "physical_strength","Coolant_temperature"
]

setting_table = {}
for code, df in train.groupby("mold_code"):
    settings = {}
    for col in setting_cols:
        smoothed = smooth_series(df[col].dropna())
        if len(smoothed) == 0:
            settings[col] = df[col].mean()
            continue
        try:
            mode_val = stats.mode(smoothed, keepdims=True)[0][0]
            settings[col] = mode_val
        except Exception:
            settings[col] = smoothed.mean()
    setting_table[str(code)] = settings  # 🔑 mold_code를 문자열로 저장

setting_df = pd.DataFrame(setting_table).T.reset_index().rename(columns={"index": "mold_code"})
setting_df["mold_code"] = setting_df["mold_code"].astype(str)  # 문자열로 통일

# ================================
# 생산 시뮬레이션 탭 비율 그래프
# ================================
train_raw = pd.read_csv("./data/train_raw.csv")

if "date" in train_raw.columns and "time" in train_raw.columns:
    train_raw["real_time"] = pd.to_datetime(
        train_raw["date"].astype(str) + " " + train_raw["time"].astype(str),
        errors="coerce"
    )
elif "registration_time" in train_raw.columns:
    train_raw["real_time"] = pd.to_datetime(train_raw["registration_time"], errors="coerce")
else:
    raise ValueError("date/time 또는 registration_time 컬럼을 확인해주세요.")

train_raw["date_only"] = train_raw["real_time"].dt.date

# 날짜별 mold_code 생산 개수
daily_mold = train_raw.groupby(["date_only", "mold_code"]).size().reset_index(name="count")
pivot_count = daily_mold.pivot(index="date_only", columns="mold_code", values="count").fillna(0)
# ================================
# 생산 시뮬레이션 탭 비율 그래프
# ================================

# ========== UI ==========
years = list(range(2024, 2027))
months = list(range(1, 13))

app_ui = ui.page_fluid(
    ui.head_content(
        ui.tags.link(rel="icon", type="image/x-icon", href="favicon.ico"),
        ui.tags.link(rel="icon", type="image/png", sizes="32x32", href="favicon-32.png"),
        ui.tags.link(rel="apple-touch-icon", sizes="180x180", href="apple-touch-icon.png"),
        ui.tags.link(rel="icon", type="image/png", sizes="192x192", href="icon-192.png"),
        ui.tags.link(rel="icon", type="image/png", sizes="512x512", href="icon-512.png"),
        ui.tags.title("주조 공정 불량 예측"),
        # Font Awesome 아이콘 불러오기
        ui.tags.link(
            rel="stylesheet",
            href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
        ),
        ui.tags.link(rel="stylesheet", type="text/css", href="custom.css"),
        # ui.tags.link(rel="stylesheet", type="text/css", href="bootstrap.css"),
        ui.tags.style("""
            /* 날짜 범위 선택 입력창 가로 배치 */
            .input-daterange {
                display: flex !important;
                align-items: center;
                gap: 6px;
            }
            .input-daterange input {
                width: 140px !important;   /* 각 칸 폭 */
            }
            .tooltip-inner {
                background-color: #ffffff !important;
                color: #000000 !important;
                border: 1px solid #ccc !important;
                max-width: 400px;   /* 툴팁이 너무 넓어지지 않게 제한 */
                text-align: left;   /* 표 왼쪽 정렬 */
            }
            .tooltip.show {
                opacity: 1 !important;   /* 툴팁이 흐려지지 않게 */
            }
        """)
    ),

    ui.HTML('<h1 class="app-title bg-primary">주조 공정 불량 예측</h1>'),

    ui.navset_tab(
        # 1. Overview
        # ===== 네비게이션 탭 =====
        ui.nav_panel("개요",
            ui.HTML("""
            <style>
                /* 카드 hover 효과 */
                .overview-card {
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                    cursor: pointer;
                    margin-top: 20px; /* 카드 살짝 아래로 이동 */
                    position: relative;
                }
                .overview-card:hover {
                    transform: scale(1.03);
                    box-shadow: 0 8px 20px rgba(0,0,0,0.15);
                }
                /* 카드 하단 [바로가기] 텍스트 */
                .card-link-text {
                    display: block;
                    text-align: center;
                    font-weight: bold;
                    font-size: 16px;
                    color: black; /* 기본 글자색 */
                    margin-top: 10px;
                    transition: color 0.3s ease;
                }
                .overview-card:hover .card-link-text {
                    color: white; /* hover 시 흰색 */
                }
            </style>

            <script>
                document.addEventListener("DOMContentLoaded", function () {
                    const cards = document.querySelectorAll(".overview-card");
                    const actions = ["go_explore", "go_predict", "go_model"];

                    cards.forEach(function(card, index) {
                        card.addEventListener("click", function() {
                            Shiny.setInputValue(actions[index], Math.random(), {priority: "event"});
                        });
                    });
                });
            </script>
            """),
            ui.layout_columns(

                # 데이터 탐색
                ui.card(
                    {"class": "overview-card", "style": "border: 2px solid #FFC966; color: #FFC966;"},
                    ui.card_header(
                        "데이터 탐색",
                        style=(
                            "background-color: #FFC966; color: #333; "
                            "font-weight:bold; font-size:20px; text-align:center; "
                            "padding-top:15px; padding-bottom:15px;"
                        )
                    ),
                    ui.tags.img(
                        src="1.png",
                        style="width:100%; height:400px; object-fit:cover; margin-bottom:10px; border-radius:8px;"
                    ),
                    ui.HTML(
                        """
                        <p style="font-size:16px; line-height:1.5; text-align:center; margin:15px 10px; color:#333;">
                            주조 공정에 대한 이해를 돕고<br>
                            사용 데이터에 대한 탐색 기능을 제공합니다.
                        </p>
                        """
                    ),
                    ui.tags.p("[바로가기]", class_="card-link-text")
                ),

                # 예측
                ui.card(
                    {"class": "overview-card", "style": "border: 2px solid #A5C16A; color: #A5C16A;"},
                    ui.card_header(
                        "예측 시뮬레이션",
                        style=(
                            "background-color: #A5C16A; color: #333; "
                            "font-weight:bold; font-size:20px; text-align:center; "
                            "padding-top:15px; padding-bottom:15px;"
                        )
                    ),
                    ui.tags.img(
                        src="3.png",
                        style="width:100%; height:400px; object-fit:cover; margin-bottom:10px; border-radius:8px;"
                    ),
                    ui.HTML(
                        """
                        <p style="font-size:16px; line-height:1.5; text-align:center; margin:15px 10px; color:#333;">
                            모델 예측 기능을 통해 입력된 공정 조건을 바탕으로<br>
                            생산품의 품질 결과 예측을 제공합니다.
                        </p>
                        """
                    ),
                    ui.tags.p("[바로가기]", class_="card-link-text")
                ),

                # 모델 학습
                ui.card(
                    {"class": "overview-card", "style": "border: 2px solid #80CBC4; color: #80CBC4;"},
                    ui.card_header(
                        "모델 학습",
                        style=(
                            "background-color: #80CBC4; color: #333; "
                            "font-weight:bold; font-size:20px; text-align:center; "
                            "padding-top:15px; padding-bottom:15px;"
                        )
                    ),
                    ui.tags.img(
                        src="2.png",
                        style="width:100%; height:400px; object-fit:cover; margin-bottom:10px; border-radius:8px;"
                    ),
                    ui.HTML(
                        """
                        <p style="font-size:16px; line-height:1.5; text-align:center; margin:15px 10px; color:#333;">
                            개발 시도한 모델 정보와<br>
                            최종 선정한 모델에 대한 설명과 근거를 제공합니다.                        
                        </p>
                        """
                    ),
                    ui.tags.p("[바로가기]", class_="card-link-text")
                ),
            )
        ),

        # 2. 데이터 탐색 (EDA)
        ui.nav_panel(
            "데이터 탐색",
            ui.navset_tab(
                ui.nav_panel(
                    "주조 공정이란?",
                        # -------------------- 상단 SVG + 버튼 --------------------
                        ui.layout_columns(
                         # 1️⃣ 왼쪽 → SVG 그림
                          ui.div(
                           ui.HTML(svg_code),
                           *[
                           ui.input_action_button(
                             f"btn_{lbl['id']}", "",
                             style=f"""
                             position:absolute;
                             top:{lbl['y']}px; left:calc(50% - 500px + {lbl['x']}px);
                             width:{lbl['w']}px; height:{lbl['h']}px;
                             opacity:0; cursor:pointer;
                             """
                          )
                         for lbl in labels
                      ],
                      style="position:relative; width:1000px; height:500px;"  
                   ),

                        #  # 2️⃣ 오른쪽 → 버튼 추가
                        ui.div(
                            [
                                ui.input_action_button(
                                    "btn_process", "[주조 공정]",
                                    style="""
                                        width:200px; margin:10px;
                                        background-color:#A5C16A;
                                        color:white; font-weight:bold; border:none;
                                        border-radius:8px;
                                    """
                                ),
                                ui.input_action_button(
                                    "btn_steps", "[주조 공정 단계]",
                                    style="""
                                        width:200px; margin:10px;
                                        background-color:#FFC966;
                                        color:white; font-weight:bold; border:none;
                                        border-radius:8px;
                                    """
                                ),
                            ],
                            style="display:flex; flex-direction:column; align-items:flex-start; justify-content:center;"
                        ),
                        col_widths=[8, 4]
                        ),


                        
                        # -------------------- JS 코드 삽입 --------------------
                        ui.tags.script("""
                            Shiny.addCustomMessageHandler("switch_tab_with_label", function(msg) {
                                let tabs = document.querySelectorAll('.nav-link');
                                tabs.forEach(function(tab) {
                                    if (tab.textContent.trim() === msg.tab) {
                                        tab.click();
                                    }
                                });
                            });
                        """),

                        # -------------------- 설명 영역 --------------------
                        ui.div(
                            {
                                "style": """
                                    margin-top:5px; 
                                    padding:5px; 
                                    border:1px solid #ddd; 
                                    border-radius:10px; 
                                    background:#fafafa;
                                """
                            },
                            ui.tags.div(
                            ui.markdown("""
                            공정 단계별 데이터를 활용하여, 주조 조건(온도, 속도, 금형, 가열로 등)이 최종 양품/불량(passorfail)에 어떤 영향을 주는지 탐색할 수 있습니다.  
                            이를 통해 주조 공정의 주요 인자들을 이해하고, **품질 개선 및 불량 감소**에 기여할 수 있는 근거를 마련할 수 있습니다.  
                            
                            **관심있는 공정을 눌러보세요!**
                            """),
                            style="line-height:1.8; font-size:16px; text-align:center; margin:5px 0;"
                            )

                            )    
                ),
                ui.nav_panel("EDA",
                    ui.layout_sidebar(
                        ui.sidebar(
                            #분포 필터
                            ui.div(
                                f"데이터 분포 그래프 필터",
                                style="background-color:#e9ecef; padding:8px 12px; border-radius:6px; text-align:center; font-weight:bold; font-size: 14px"
                            ),
                            ui.input_selectize(
                                "mold_code2",
                                "Mold Code 선택",
                                choices=list(map(str, sorted(df_explore["mold_code"].dropna().unique())))
                            ),
                            ui.input_select(
                                "var",
                                "분석 변수 선택",
                                choices={c: get_label(c) for c in df_explore.columns if c not in ["mold_code", "passorfail"]}
                            ),
                            ui.output_ui("filter_ui"),   # ★ 선택된 변수에 맞는 필터 UI
                        ),
                        ui.card(
                            ui.card_header("데이터 분포"),
                                ui.output_plot("dist_plot"),
                        ),
                    ),
                    ui.layout_sidebar(
                        ui.sidebar(
                            # 시계열 필터
                            ui.div(
                                f"시계열 데이터 필터",
                                style="background-color:#e9ecef; padding:8px 12px; border-radius:6px; text-align:center; font-weight:bold;"
                            ),
                            ui.input_select(
                                "ts_var", "원본 변수 선택",
                                choices=["없음"] + [get_label(c) for c in df_explore.columns 
                                                   if c not in ["id","line","name","mold_name","date","time","registration_time","passorfail",
                                                                "speed_ratio","pressure_speed_ratio"]],
                                selected="없음"
                            ),
                            ui.input_select(
                                "ts_var_derived", "파생 변수 선택",
                                choices=["없음"] + [get_label(c) for c in ["speed_ratio","pressure_speed_ratio"] if c in df_explore.columns],
                                selected="없음"
                            ),
                            ui.output_ui("ts_filter_ui")   # 시계열 전용 시간 필터
                        ),
                        ui.card(
                            ui.card_header("시계열 데이터"),
                                output_widget("timeseries_plot")
                        ),
                    ),
                    
                    # 2️⃣ 새로운 Boxplot 비교 카드 (독립적)
                    ui.layout_sidebar(
                        ui.sidebar(
                            ui.div(
                                "Boxplot 비교 필터",
                                style="background-color:#e9ecef; padding:8px 12px; border-radius:6px; text-align:center; font-weight:bold;"
                            ),
                            ui.input_select(
                                "box_var", "원본 변수 선택",
                                choices=["없음"] + [get_label(c) for c in df_explore.columns 
                                                   if c not in ["id","line","name","mold_name","date","time","registration_time","passorfail",
                                                                "speed_ratio","pressure_speed_ratio"]],
                                selected="없음"
                            ),
                            ui.input_select(
                                "box_var_derived", "파생 변수 선택",
                                choices=["없음"] + [get_label(c) for c in ["speed_ratio","pressure_speed_ratio"] if c in df_explore.columns],
                                selected="없음"
                            ),
                        ),
                        ui.layout_columns(
                            ui.card(
                                ui.card_header("원본 Boxplot"),
                                ui.output_plot("boxplot_raw")   # ✅ 변경
                            ),
                            ui.card(
                                ui.card_header("전처리 Boxplot"),
                                ui.output_plot("boxplot_proc")  # ✅ 변경
                            ),
                            col_widths=[6, 6]
                        ),
                    ),
                    # ui.layout_columns(
                        # # 1행
                        # ui.card(
                        #     ui.card_header("데이터 요약"),
                        #     ui.output_table("df_summary"),
                        # ),
                        # ui.card(
                        #     ui.card_header("컬럼별 결측치 비율"),
                        #     ui.output_plot("missing_plot"),
                        # ),
                        # # 2행
                        # ui.card(
                        #     ui.card_header("변수 타입 분포"),
                        #     ui.output_plot("dtype_pie"),
                        # ),
                        # ui.card(
                        #     ui.card_header("수치형 변수 상관관계"),
                        #     ui.output_plot("corr_heatmap_overview"),
                        # ),
                        # col_widths=[6, 6],  # 2열 레이아웃
                    # ),
                )
            )
        ),

        # # 3. 전처리 과정
        # ui.nav_panel(
        #     "전처리",
        #     # ui.card(ui.card_header("결측치 처리 전/후 비교"), ui.output_plot("preprocess_plot")),
        #     # ui.card(ui.card_header("이상치 처리 결과"), ui.output_plot("outlier_plot"))
        # ),

        # 5. 예측
        ui.nav_panel(
            "예측 시뮬레이션",
            ui.navset_tab(
                ui.nav_panel("예측",
                    # 입력 변수 카드
                    ui.div(
                        ui.card(
                            ui.card_header("입력 변수", style="background-color:#f8f9fa; text-align:center;"),
                            # 생산 환경 정보 카드 (최상단)
                            ui.card(
                                ui.card_header("생산 환경 정보", style="text-align:center;"),
                                ui.layout_columns(
                                    ui.div(
                                        f"생산 라인: {df_raw['line'].iloc[0]}",
                                        style="background-color:#e9ecef; padding:8px 12px; border-radius:6px; text-align:center; font-weight:bold;"
                                    ),
                                    ui.div(
                                        f"장비 이름: {df_raw['name'].iloc[0]}",
                                        style="background-color:#e9ecef; padding:8px 12px; border-radius:6px; text-align:center; font-weight:bold;"
                                    ),
                                    ui.div(
                                        f"금형 이름: {df_raw['mold_name'].iloc[0]}",
                                        style="background-color:#e9ecef; padding:8px 12px; border-radius:6px; text-align:center; font-weight:bold;"
                                    ),
                                    col_widths=[4,4,4]
                                )
                            ),

                            # === 공정 상태 관련 (4열) ===
                            ui.card(
                                ui.card_header("공정 상태 관련", style=""),
                                ui.layout_columns(
                                    ui.input_numeric("count", "일조 누적 제품 개수", value=int(df_predict["count"].mean())),
                                    ui.input_numeric("monthly_count", "월간 누적 제품 개수", value=int(df_predict["monthly_count"].mean())),
                                    ui.input_numeric("global_count", "전체 누적 제품 개수", value=int(df_predict["global_count"].mean())),
                                    ui.input_numeric("speed_ratio", "상하 구역 속도 비율", value=int(df_predict["speed_ratio"].mean())),
                                    ui.input_numeric("pressure_speed_ratio", "주조 압력 속도 비율", value=int(df_predict["pressure_speed_ratio"].mean())),
                                    make_select("working", "장비 가동 여부"),
                                    make_select("emergency_stop", "비상 정지 여부"),
                                    make_select("tryshot_signal", "측정 딜레이 여부"),
                                    make_select("shift", "주, 야간 조"),
                                    col_widths=[3,3,3,3]
                                )
                            ),

                            # === 용융 단계 (n행 4열) ===
                            ui.card(
                                ui.card_header("용융 단계", style=""),
                                ui.layout_columns(
                                    make_num_slider("molten_temp"),
                                    make_select("heating_furnace", "용해로"),
                                    col_widths=[6,6]
                                )
                            ),

                            # === 충진 단계 (n행 4열) ===
                            ui.card(
                                ui.card_header("충진 단계", style=""),
                                ui.layout_columns(
                                    make_num_slider("sleeve_temperature"),
                                    make_num_slider("EMS_operation_time"),
                                    make_num_slider("low_section_speed"),
                                    make_num_slider("high_section_speed"),
                                    make_num_slider("molten_volume"),
                                    make_num_slider("cast_pressure"),
                                    ui.input_select("mold_code", "금형 코드", choices=sorted(df_predict["mold_code"].dropna().unique().astype(str))),
                                    col_widths=[3,3,3,3]
                                )
                            ),

                            # === 냉각 단계 (n행 4열) ===
                            ui.card(
                                ui.card_header("냉각 단계", style=""),
                                ui.layout_columns(
                                    make_num_slider("upper_mold_temp1"),
                                    make_num_slider("upper_mold_temp2"),
                                    make_num_slider("upper_mold_temp3"),
                                    make_num_slider("lower_mold_temp1"),
                                    make_num_slider("lower_mold_temp2"),
                                    make_num_slider("lower_mold_temp3"),
                                    make_num_slider("Coolant_temperature"),
                                    col_widths=[3,3,3,3]
                                )
                            ),

                            # === 공정 속도 관련 (n행 4열) ===
                            ui.card(
                                ui.card_header("공정 속도 관련", style=""),
                                ui.layout_columns(
                                    make_num_slider("facility_operation_cycleTime"),
                                    make_num_slider("production_cycletime"),
                                    col_widths=[6,6]
                                )
                            ),

                            # === 품질 및 성능 (n행 4열) ===
                            ui.card(
                                ui.card_header("품질 및 성능", style=""),
                                ui.layout_columns(
                                    make_num_slider("biscuit_thickness"),
                                    make_num_slider("physical_strength"),
                                    col_widths=[6,6]
                                )
                            )
                        ),
                        style="max-width: 1200px; margin: 0 auto;"
                    ),

                    ui.br(),

                    # 예측 실행 + 결과 카드 (sticky)
                    ui.div(
                        ui.card(
                            ui.card_header(
                                ui.div(
                                    [
                                        ui.input_action_button(
                                            "predict_btn", "예측 실행",
                                            class_="btn btn-primary btn-lg",
                                            style="flex:1;"
                                        ),
                                        ui.input_action_button(
                                            "reset_btn", ui.HTML('<i class="fa-solid fa-rotate-left"></i>'),
                                            class_="btn btn-secondary btn-lg",
                                            style="margin-left:10px; width:60px;"
                                        )
                                    ],
                                    style="display:flex; align-items:center; width:100%;"
                                ),
                                style="background-color:#f8f9fa; text-align:center;" 
                            ),
                            ui.output_ui("prediction_result")
                        ),
                        style="""
                            position: -webkit-sticky;
                            position: sticky;
                            bottom: 1px;
                            z-index: 1000;
                            max-width: 1200px;
                            margin: 0 auto;
                        """
                    ),

                ),
                ui.nav_panel("개선 방안",
                    ui.card(
                        ui.card_header("불량 기여 요인 Top 5", style="text-align:center;"),
                        ui.output_plot("local_factor_plot"),
                        ui.hr(),
                        ui.output_ui("local_factor_desc")   # ← 설명 칸 추가
                    )
                ),
                ui.nav_panel(
                    "생산계획",
                    ui.layout_sidebar(
                        ui.sidebar(
                            ui.input_numeric("monthly_target", "이달의 총 생산 목표 수",
                                            value=20000, min=1000, step=1000),
                            ui.input_select("year", "연도 선택", {str(y): str(y) for y in years},
                                            selected=str(datetime.date.today().year)),
                            ui.input_select("month", "월 선택", {str(m): f"{m}월" for m in months},
                                            selected=str(datetime.date.today().month)),
                            ui.output_ui("mold_inputs"),
                            ui.output_text("remaining_qty"),
                            ui.input_action_button("run_plan", "시뮬레이션 실행", class_="btn btn-primary"),
                        ),
                        ui.card(
                            ui.card_header("금형코드별 생산성 요약"),
                            ui.output_data_frame("mold_summary_table"),
                            style="flex: 0 0 auto;"
                        ),
                        ui.card(
                            ui.card_header("달력형 계획표",
                                ui.input_action_button("show_modal", "날짜별 금형 코드 생산 추이", class_="btn btn-primary", 
                                    style="position:absolute; top:10px; right:10px; height:30px; font-size:12px; display:flex; align-items:center; justify-content:center;"
                                ),
                            ),
                            ui.output_ui("calendar_view"),
                            ui.p(
                                "※ 몰드코드에 따른 공정 조건을 확인하세요!", 
                                style="font-size:15px; color:gray; margin-top:4px;"
                            ),
                            ui.p(
                                "※ 선택한 연월의 금형 계획과 공정 조건을 확인 가능 합니다. 몰드별 최대 생산량을 고려한 조건임을 유의하세요.", 
                                style="font-size:15px; color:gray; margin-top:4px;"
                            )
                        )
                    )
                )
            )
        ),
# 4. 모델 학습
ui.nav_panel(
    "모델 학습",
    ui.navset_tab(

        # ============================
        # 서브탭 1: 모델 선택
        # ============================
        ui.nav_panel(
            "모델 선택",
            ui.div(
                [
                    # 🔹 첫 줄: 혼동 행렬 3개
                    ui.layout_columns(
                        ui.card(
                            ui.card_header(
                                ui.div(
                                    [
                                        "Random Forest",
                                        ui.tooltip(
                                            ui.tags.i(
                                                class_="fa-solid fa-circle-info",
                                                style="color:#007bff; cursor:pointer; margin-left:6px;"
                                            ),
                                            ui.HTML("""
                                                <p style="font-size:13px; font-weight:bold; margin-top:0; margin-bottom:6px;">
                                                    혼동행렬 설명
                                                </p>
                                                <table border="1" style="border-collapse:collapse; font-size:12px; text-align:center;">
                                                    <thead style="background:#f9f9f9;">
                                                        <tr>
                                                            <th></th>
                                                            <th>Pred: 불량</th>
                                                            <th>Pred: 정상</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        <tr>
                                                            <th style="background:#f9f9f9;">Actual: 불량</th>
                                                            <td style="color:green; font-weight:bold;">TP</td>
                                                            <td style="color:red; font-weight:bold;">FN</td>
                                                        </tr>
                                                        <tr>
                                                            <th style="background:#f9f9f9;">Actual: 정상</th>
                                                            <td style="color:red; font-weight:bold;">FP</td>
                                                            <td style="color:green; font-weight:bold;">TN</td>
                                                        </tr>
                                                    </tbody>
                                                </table>
                                                <p style="font-size:12px; margin-top:6px; text-align:left;">
                                                    - <b>TP</b>: 실제 불량을 불량으로 예측<br>
                                                    - <b>FN</b>: 실제 불량을 정상으로 잘못 예측<br>
                                                    - <b>FP</b>: 실제 정상을 불량으로 잘못 예측<br>
                                                    - <b>TN</b>: 실제 정상을 정상으로 예측
                                                </p>
                                            """),
                                            placement="right"
                                        )
                                    ],
                                    style="display:flex; align-items:center;"
                                )
                            ),
                            ui.output_plot("rf_cm")
                        ),
                        ui.card(
                            ui.card_header(
                                ui.div(
                                    [
                                        "LightGBM",
                                        ui.tooltip(
                                            ui.tags.i(
                                                class_="fa-solid fa-circle-info",
                                                style="color:#007bff; cursor:pointer; margin-left:6px;"
                                            ),
                                            ui.HTML("""
                                                <p style="font-size:13px; font-weight:bold; margin-top:0; margin-bottom:6px;">
                                                    혼동행렬 설명
                                                </p>
                                                <table border="1" style="border-collapse:collapse; font-size:12px; text-align:center;">
                                                    <thead style="background:#f9f9f9;">
                                                        <tr>
                                                            <th></th>
                                                            <th>Pred: 불량</th>
                                                            <th>Pred: 정상</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        <tr>
                                                            <th style="background:#f9f9f9;">Actual: 불량</th>
                                                            <td style="color:green; font-weight:bold;">TP</td>
                                                            <td style="color:red; font-weight:bold;">FN</td>
                                                        </tr>
                                                        <tr>
                                                            <th style="background:#f9f9f9;">Actual: 정상</th>
                                                            <td style="color:red; font-weight:bold;">FP</td>
                                                            <td style="color:green; font-weight:bold;">TN</td>
                                                        </tr>
                                                    </tbody>
                                                </table>
                                                <p style="font-size:12px; margin-top:6px; text-align:left;">
                                                    - <b>TP</b>: 실제 불량을 불량으로 예측<br>
                                                    - <b>FN</b>: 실제 불량을 정상으로 잘못 예측<br>
                                                    - <b>FP</b>: 실제 정상을 불량으로 잘못 예측<br>
                                                    - <b>TN</b>: 실제 정상을 정상으로 예측
                                                </p>
                                            """),
                                            placement="right"
                                        )
                                    ],
                                    style="display:flex; align-items:center;"
                                )
                            ),
                            ui.output_plot("lgbm_cm")
                        ),
                        ui.card(
                            ui.card_header(
                                ui.div(
                                    [
                                        "XGBoost",
                                        ui.tooltip(
                                            ui.tags.i(
                                                class_="fa-solid fa-circle-info",
                                                style="color:#007bff; cursor:pointer; margin-left:6px;"
                                            ),
                                            ui.HTML("""
                                                <p style="font-size:13px; font-weight:bold; margin-top:0; margin-bottom:6px;">
                                                    혼동행렬 설명
                                                </p>
                                                <table border="1" style="border-collapse:collapse; font-size:12px; text-align:center;">
                                                    <thead style="background:#f9f9f9;">
                                                        <tr>
                                                            <th></th>
                                                            <th>Pred: 불량</th>
                                                            <th>Pred: 정상</th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        <tr>
                                                            <th style="background:#f9f9f9;">Actual: 불량</th>
                                                            <td style="color:green; font-weight:bold;">TP</td>
                                                            <td style="color:red; font-weight:bold;">FN</td>
                                                        </tr>
                                                        <tr>
                                                            <th style="background:#f9f9f9;">Actual: 정상</th>
                                                            <td style="color:red; font-weight:bold;">FP</td>
                                                            <td style="color:green; font-weight:bold;">TN</td>
                                                        </tr>
                                                    </tbody>
                                                </table>
                                                <p style="font-size:12px; margin-top:6px; text-align:left;">
                                                    - <b>TP</b>: 실제 불량을 불량으로 예측<br>
                                                    - <b>FN</b>: 실제 불량을 정상으로 잘못 예측<br>
                                                    - <b>FP</b>: 실제 정상을 불량으로 잘못 예측<br>
                                                    - <b>TN</b>: 실제 정상을 정상으로 예측
                                                </p>
                                            """),
                                            placement="right"
                                        )
                                    ],
                                    style="display:flex; align-items:center;"
                                )
                            ),
                            ui.output_plot("xgb_cm")
                        ),
                        col_widths=[4,4,4]
                    ),

                    ui.br(),

                    # 🔹 두 번째 줄: Best Score 카드
                    ui.card(
                        ui.card_header(
                            ui.div(
                                [
                                    "📊 Best Score 결과",
                                    ui.tooltip(
                                        ui.tags.i(
                                            class_="fa-solid fa-circle-info",
                                            style="color:#007bff; cursor:pointer; margin-left:8px;"
                                        ),
                                        ui.HTML("""
                                            <p style="font-size:13px; font-weight:bold; margin-top:0; margin-bottom:6px;">
                                                Best Score 계산 공식
                                            </p>
                                            <p style="font-size:13px; line-height:1.6; text-align:left;">
                                                Best Score = (0.1 × Accuracy) + (0.6 × Recall) + (0.3 × F1)<br><br>
                                                - Accuracy (ACC): 전체 데이터 중 예측이 맞은 비율<br>
                                                - Recall: 실제 불량 중에서 불량으로 올바르게 잡아낸 비율<br>
                                                - F1 Score: Precision과 Recall의 조화 평균<br><br>
                                                ※ 불량 검출 중요도를 높이기 위해 <b>Recall 가중치(0.6)</b>를 가장 크게 부여했습니다.
                                            </p>
                                        """),
                                        placement="right"
                                    )
                                ],
                                style="display:flex; align-items:center;"
                            )
                        ),
                        ui.div(
                            [
                                # 표
                                ui.HTML(""" 
                                    <div style="padding:10px; font-size:15px;">
                                        <p><b>가중치: ACC(0.1), Recall(0.6), F1(0.3)</b></p>
                                        <table border="1" style="border-collapse:collapse; width:100%; text-align:center;">
                                            <thead style="background:#f0f0f0;">
                                                <tr>
                                                    <th>Model</th>
                                                    <th>ACC</th>
                                                    <th>Recall</th>
                                                    <th>F1</th>
                                                    <th>Best Score</th>
                                                    <th>Rank</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                <tr>
                                                    <td>XGBoost</td>
                                                    <td>0.990</td>
                                                    <td>0.978</td>
                                                    <td>0.923</td>
                                                    <td>0.9627</td>
                                                    <td>🥇 1위</td>
                                                </tr>
                                                <tr>
                                                    <td>LightGBM</td>
                                                    <td>0.994</td>
                                                    <td>0.968</td>
                                                    <td>0.930</td>
                                                    <td>0.9592</td>
                                                    <td>🥈 2위</td>
                                                </tr>
                                                <tr>
                                                    <td>RandomForest</td>
                                                    <td>0.990</td>
                                                    <td>0.976</td>
                                                    <td>0.899</td>
                                                    <td>0.9543</td>
                                                    <td>🥉 3위</td>
                                                </tr>
                                            </tbody>
                                        </table>
                                        <br>
                                    </div>
                                """),

                                # 그래프 출력
                                ui.output_plot("best_score_plot")
                            ]
                        )
                    )

                ]
            )
        ),

        # ============================
        # 서브탭 2: 모델 설명
        # ============================
        ui.nav_panel(
            "모델 설명",
            ui.div(
                [
                    # 🔹 1. XGBoost 선택 이유 (상단 공통 카드)
                    ui.card(
                        ui.card_header("왜 XGBoost를 최종 모델로 선택했는가?"),


                        ui.input_action_button(
                            "help_btn",  # 버튼 ID
                            ui.HTML('<i class="fa-solid fa-circle-question fa-lg" style="color:#007bff;"></i>'),
                            class_="btn btn-link",
                            style="position:absolute; top:10px; right:10px;"  # 카드 오른쪽 위에 고정
                        ),



                        ui.HTML("""
                            <ul style="font-size:15px; line-height:1.8; text-align:left;">
                                <li><b>불량 검출 (Recall) 최우선</b>: XGBoost Recall 97.8% (세 모델 중 최고)</li>
                                <li><b>균형 잡힌 성능</b>: Recall 97.8% + F1 92.3% → 불량 검출력과 비용 최소화 동시 달성</li>
                                <li><b>안정성과 재현성</b>: Trial 반복 실험에서도 상위권 유지 → 안정적, 신뢰성 확보</li>
                                <li><b>경제적 효과</b>: Recall↑ → 재작업/폐기 비용 절감, Precision 유지 → 불필요 검사 비용 절감</li>
                            </ul>
                        """)
                    ),
        
                    ui.br(),
        
                    # 🔹 2. Feature Importance & SHAP 그래프 (좌우 배치)
                    ui.layout_columns(
                    
                        # 왼쪽: Feature Importance
                        ui.card(
                            ui.card_header("Feature Importance (중요 변수 Top)"),
                            ui.tags.img(
                                src="feature_importance.png",   # www 폴더 안에 넣어두기
                                style="width:100%; height:auto; border-radius:6px; margin-bottom:10px;"
                            ),
                            ui.HTML("""
                                <p style="font-size:14px; text-align:left; line-height:1.6; margin:10px;">
                                    - 월간 생산 횟수, 슬리브 온도, 금형 온도 변수가 가장 큰 영향을 미침<br>
                                    - 해당 다이캐스팅 공정 환경에서 불량 판정의 주요 기준이 <b>온도·횟수·압력</b>임을 확인<br>
                                    - 생산 현장에서 <b>우선적으로 관리가 필요한 변수 순위</b>를 직관적으로 확인 가능<br>
                                </p>
                            """)
                        ),
        
                        # 오른쪽: SHAP Summary Plot
                        ui.card(
                            ui.card_header("SHAP Summary Plot (변수 영향 방향성)"),
                            ui.tags.img(
                                src="shap_summary.png",
                                style="max-width:550px; height:auto; border-radius:6px; margin-bottom:10px;"
                            ),
                            ui.HTML("""
                                <p style="font-size:14px; text-align:left; line-height:1.6; margin:10px;">
                                    - <b>압력/속도 비율이 증가</b>할수록 불량률 증가: 과도한 압력/속도 불균형 문제 발생<br>
                                    - <b>생산 횟수가 증가</b>할수록 불량 감소: 안정화 효과<br>
                                    - 데이터 기반으로 <b>세부 공정 개선 방안 제시를 통한 공정 효율 극대화</b> 가능
                                </p>
                            """)
                        ),
        
                        col_widths=[6,6]
                    )
                ]
            )
        )
    )
),

        id="main_nav",   # ⭐ 탭 컨트롤을 위한 id
    )
)

# ===== 대표 양품행 만드는 함수 =====
def get_representative_good(X_row, n_neighbors=3):
    # 사용 가능한 수치형/범주형 분리
    exclude_vars = ["count", "monthly_count", "global_count"]
    use_num_cols = [c for c in num_cols if c not in exclude_vars]

    good_df = df_predict[df_predict["passorfail"] == 0][use_num_cols + cat_cols].dropna()

    # --- 수치형 거리 계산 ---
    num_dist = pairwise_distances(
        X_row[use_num_cols], good_df[use_num_cols], metric="euclidean"
    )[0]

    # 가까운 N개 index
    idx = np.argsort(num_dist)[:n_neighbors]
    nearest = good_df.iloc[idx]

    # 대표행 만들기: 수치형 = 평균, 범주형 = 최빈값
    rep = {}
    for col in use_num_cols:
        rep[col] = nearest[col].mean()
    for col in cat_cols:
        rep[col] = nearest[col].mode().iloc[0]

    return pd.DataFrame([rep])
      
# ===== SERVER (변경 없음) =====
def server(input, output, session):
    #====== 개요에서 카드 클릭 시 탭이동 =================================
    @reactive.Effect
    @reactive.event(input.go_explore)
    def _():
        update_navs("main_nav", selected="데이터 탐색")

    @reactive.Effect
    @reactive.event(input.go_predict)
    def _():
        update_navs("main_nav", selected="예측 시뮬레이션")

    @reactive.Effect
    @reactive.event(input.go_model)
    def _():
        update_navs("main_nav", selected="모델 학습")
    #=================================================================
    
    # 원본 변수 선택 시 → 파생 변수 '없음'으로 자동 변경
    @reactive.effect
    @reactive.event(input.ts_var)
    def _():
        if input.ts_var() != "없음":
            ui.update_select("ts_var_derived", selected="없음")

    # 파생 변수 선택 시 → 원본 변수 '없음'으로 자동 변경
    @reactive.effect
    @reactive.event(input.ts_var_derived)
    def _():
        if input.ts_var_derived() != "없음":
            ui.update_select("ts_var", selected="없음")

    # 서버 함수 안에
    @reactive.effect
    @reactive.event(input.goto_explore)
    def _():
        ui.update_navs("main_tabs", selected="데이터 탐색")

    @reactive.effect
    @reactive.event(input.goto_preprocess)
    def _():
        ui.update_navs("main_tabs", selected="전처리")

    @reactive.effect
    @reactive.event(input.goto_train)
    def _():
        ui.update_navs("main_tabs", selected="모델 학습")

    @reactive.effect
    @reactive.event(input.goto_predict)
    def _():
        ui.update_navs("main_tabs", selected="예측")

    # 버튼 클릭 시 탭 전환
    for lbl in labels:
        @reactive.Effect
        @reactive.event(getattr(input, f"btn_{lbl['id']}"))
        async def _(lbl=lbl):
            # 1️⃣ 탭 전환 (비동기 → await 필요)
            await session.send_custom_message("switch_tab_with_label", {
                "tab": "EDA",
                "label": lbl["var"]
            })
            # 2️⃣ 드롭다운 선택값 업데이트 (동기 → await 쓰면 안됨)
            session.send_input_message(
                "var",
                update_select("var", selected=lbl["var"])
            )
            session.send_input_message(
                "ts_var",
                update_select("ts_var", selected=lbl["var"])
            )

    # ===== 모델 학습 - 혼동 행렬 =====
    conf_matrices = {
        "Random Forest": [[488, 12], [88, 9412]],
        "LightGBM": [[484, 16], [44, 9456]],
        "XGBoost": [[489, 11], [89, 9411]],
    }

    def plot_confusion_matrix(matrix, title):
        cm = [[matrix[0][0], matrix[0][1]],   # 실제 불량 (TP, FN)
              [matrix[1][0], matrix[1][1]]]   # 실제 정상 (FP, TN)

        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", cbar=False, ax=ax,
                    xticklabels=["Pred: 불량", "Pred: 정상"],
                    yticklabels=["Actual: 불량", "Actual: 정상"])
        ax.set_title(title)
        plt.tight_layout()
        return fig

    @output
    @render.plot
    def rf_cm():
        return plot_confusion_matrix(conf_matrices["Random Forest"], "Random Forest")

    @output
    @render.plot
    def lgbm_cm():
        return plot_confusion_matrix(conf_matrices["LightGBM"], "LightGBM")

    @output
    @render.plot
    def xgb_cm():
        return plot_confusion_matrix(conf_matrices["XGBoost"], "XGBoost")

    # Best Score 데이터
    df_scores = pd.DataFrame({
        "Model": ["XGBoost", "LightGBM", "RandomForest"],
        "BestScore": [0.9627, 0.9592, 0.9543]
    })

    @output
    @render.plot
    def best_score_plot():
        fig, ax = plt.subplots(figsize=(6,4))
        sns.barplot(data=df_scores, x="Model", y="BestScore", palette="Oranges_r", ax=ax)

        # 점수 표시
        for i, row in df_scores.iterrows():
            ax.text(i, row["BestScore"] + 0.0003, f"{row['BestScore']:.4f}", 
                    ha="center", fontsize=10)

        ax.set_title("Model Best Score Ranking (ACC 0.1, Recall 0.6, F1 0.3)", fontsize=12)
        ax.set_ylabel("Best Score")
        ax.set_ylim(0.953, 0.964)
        plt.tight_layout()
        return fig

    @output
    @render.text
    def selected_var():
        return f"현재 선택된 변수: {input.var() or '없음'}"

    last_proba = reactive.value(None)
    loading = reactive.value(False)
    local_factors = reactive.value(None)
    
    def get_input_data():
        data = {}
        for col in cat_cols + num_cols:
            data[col] = [input[col]()]

        return pd.DataFrame(data)

    for col in num_cols:
        @reactive.effect
        @reactive.event(input[col])
        def _(col=col):
            update_slider(f"{col}_slider", value=input[col]())
        @reactive.effect
        @reactive.event(input[f"{col}_slider"])
        def _(col=col):
            update_numeric(col, value=input[f"{col}_slider"]())

    @reactive.effect
    @reactive.event(input.reset_btn)
    def _():
        # 범주형 변수: 첫 번째 값으로 초기화
        for col in cat_cols:
            first_val = str(sorted(df_predict[col].dropna().unique())[0])
            if(col == "tryshot_signal"):
                first_val = "없음"
            ui.update_select(col, selected=first_val)

        # 수치형 변수: 안전하게 숫자 변환 후 평균값으로 초기화
        for col in num_cols:
            series = pd.to_numeric(df_predict[col], errors="coerce")       # 문자열 → 숫자 (에러시 NaN)
            series = series.replace([np.inf, -np.inf], np.nan)             # inf → NaN
            mean_val = series.dropna().mean()                              # NaN 제거 후 평균
            default_val = int(mean_val) if pd.notna(mean_val) else 0       # fallback: 0
            update_slider(f"{col}_slider", value=default_val)
            update_numeric(col, value=default_val)

        # 예측 결과 초기화
        last_proba.set(None)
    # [주조 공정] 버튼 클릭 → 팝업
    @reactive.effect
    @reactive.event(input.btn_process)   # 이름 맞춤
    def _():
        ui.modal_show(
            ui.modal(
                ui.markdown("""
                **주조(Casting) 공정**  

                주조(Casting)는 금속을 녹여 원하는 형상을 만드는 제조 공정입니다.  
                고체 상태의 금속을 고온에서 녹여 액체 상태로 만든 뒤, 미리 준비된 금형에 부어 응고시키면 제품 형태가 완성됩니다.  

                주조 공정은 복잡한 형상, 대량 생산, 재료 절감이 가능하여 자동차, 기계 부품 등 다양한 산업 분야에서 널리 활용됩니다.  

                ---

                **주요 목적**  
                - 금속을 원하는 형상과 치수로 성형  
                - 기계적 강도와 품질 확보  
                - 공정 효율 및 생산성 향상  

                ---

                **다이캐스팅 공정 (Die Casting)**  
                
                다이캐스팅은 고압을 이용해 용융 금속을 금형 내로 빠르게 주입하여, 복잡한 형상을 가진 금속 부품을 고속으로 생산하는 공정입니다.  
                정밀한 치수, 매끄러운 표면, 높은 생산성을 달성할 수 있는 것이 특징입니다.  
                """),
                title="주조 공정 설명",
                easy_close=True,
                footer=ui.modal_button("닫기"),
                size="xl"
            )
        )

    # [주조 공정 단계] 버튼 클릭 → 팝업
    @reactive.effect
    @reactive.event(input.btn_steps)   # 이름 맞춤
    def _():
        # ui.modal_show(
        ui.modal_show(
            ui.modal(
                ui.markdown("""
                **1. 용융 단계 (Melting)**  

                금속을 고온에서 녹이는 과정입니다.  
                용해로를 통해 일정 온도로 금속을 유지하며, 주입 가능한 액체 상태를 만듭니다.  
                이 단계에서 금속의 균질성과 온도 관리가 매우 중요합니다.  

                ---

                **2. 충진 단계 (Filling)**  

                녹인 금속을 금형 내부로 주입하는 단계입니다.  
                주입 속도, 주입 압력, 금형 설계에 따라 내부 충진 상태가 달라지고, 제품 내부 결함(공극, 불균질 등)에 영향을 줍니다.  
                일부 공정에서는 전자 교반(EMS)을 통해 금속 혼합을 개선하기도 합니다.  

                ---

                **3. 냉각 단계 (Cooling)**  

                주입된 금속이 금형 내에서 응고되는 단계입니다.  
                금속의 냉각 속도와 금형 온도를 적절히 제어해야 수축, 변형, 내부 응력 등을 최소화할 수 있습니다.  
                냉각수와 금형 온도 관리가 주요 역할을 합니다.  

                ---

                **4. 공정 속도 및 장비 운전**  
                
                장비 사이클 시간과 실제 생산 속도는 공정 효율과 품질 안정성에 직결됩니다.  
                장비 가동 상태, 비상 정지 여부 등을 관리하며 생산 계획에 따라 운용됩니다.  

                ---

                **5. 품질 평가 (Inspection)**  
                
                최종 주조물은 두께, 강도 등 물리적 특성을 평가합니다.  
                합격/불합격(pass/fail) 여부를 결정하며, 이를 기반으로 공정 최적화와 품질 개선을 수행합니다.  
                """),
                title="주조 공정 단계",
                easy_close=True,
                footer=ui.modal_button("닫기"),
                size="xl"
            )
        )
    @reactive.effect
    @reactive.event(input.help_btn)
    def _():
     ui.modal_show(
    ui.modal(
        ui.div(
            [
                ui.markdown("모델 학습 과정을 도와드리겠습니다."),
                 ui.div(
                        ui.tags.img(
                            src="help.jpg",
                            style="width:300px; border-radius:10px;"
                        ),
                        style="text-align:center; margin-top:15px;"
                 )
            ]
        ),
        title=ui.HTML("<b>📘 모델 학습 도우미</b>"),
        easy_close=True,
        footer=ui.modal_button("010-4377-9710 연락주세요")
    )
)
    
    @reactive.effect
    @reactive.event(input.predict_btn)
    def _():
     loading.set(True)
     try:
        X = get_input_data()
        proba = model.predict_proba(X)[0, 1]
        last_proba.set(proba)

        # === 불량 기여 요인 계산 ===
        # 1) 누적형 변수 제거
        exclude_vars = ["count", "monthly_count", "global_count"]
        use_num_cols = [c for c in num_cols if c not in exclude_vars]

        baseline = df_predict[df_predict["passorfail"] == 0][use_num_cols].mean()
        current = X[use_num_cols].iloc[0]

        # 2) 표준화 거리 (표준편차로 나눔)
        stds = df_predict[use_num_cols].std().replace(0, 1)  # 분모=0 방지
        diffs = ((current - baseline) / stds) ** 2

        # 3) 기여도 계산
        if diffs.sum() > 0:
            contrib = (diffs / diffs.sum()).sort_values(ascending=False)
            local_factors.set(
                pd.DataFrame({
                    "feature": [get_label(c) for c in contrib.index],
                    "importance": contrib.values
                })
            )
        else:
            local_factors.set(
                pd.DataFrame({"feature": [], "importance": []})
            )

     except Exception as e:
        last_proba.set(f"error:{e}")
     finally:
        loading.set(False)
    
    @reactive.effect
    @reactive.event(input.apply_suggestions)
    def _():
        factors = local_factors()
        if factors is None or factors.empty:
            return

        top = factors.head(5).copy()
        exclude_vars = ["count", "monthly_count", "global_count"]
        use_num_cols = [c for c in num_cols if c not in exclude_vars]

        baseline = df_predict[df_predict["passorfail"] == 0][use_num_cols].mean()
        current = get_input_data().iloc[0][use_num_cols]

        for _, row in top.iterrows():
            feat = row["feature"]
            col = [k for k, v in label_map.items() if v == feat]
            if not col: 
                continue
            col = col[0]

            if col in current.index:
                diff = current[col] - baseline[col]
                if abs(diff) > 1e-6:
                    new_val = current[col] - diff/2   # 현재값과 baseline 사이 중간으로 이동
                    update_slider(f"{col}_slider", value=float(new_val))
                    update_numeric(col, value=float(new_val))
                    print(f"[반영됨] {col}: {current[col]} → {new_val} (baseline {baseline[col]})")

        # 🔹 자동 예측 실행
        session.send_input_message("predict_btn", 1)

    @render.ui
    def prediction_result():
        if loading():
            return ui.div(
                ui.div(class_="spinner-border text-primary", role="status"),
                ui.HTML("<div style='margin-top:10px;'>예측 실행 중...</div>"),
                style="text-align:center; padding:20px;"
            )

        proba = last_proba()
        if proba is None:
            return ui.div(
                ui.HTML("<span style='color:gray; font-size:18px;'>아직 예측을 실행하지 않았습니다.</span>"),
                style="text-align:center; padding:20px;"
            )

        if isinstance(proba, str) and proba.startswith("error:"):
            return ui.div(
                ui.HTML(f"<span style='color:red;'>예측 중 오류 발생: {proba[6:]}</span>")
            )

        if proba < 0.02:
            style = "background-color:#d4edda; color:#155724; font-size:18px; font-weight:bold; padding:15px; text-align:center; border-radius:8px;"
        elif proba < 0.04:
            style = "background-color:#fff3cd; color:#856404; font-size:18px; font-weight:bold; padding:15px; text-align:center; border-radius:8px;"
        else:
            style = "background-color:#f8d7da; color:#721c24; font-size:18px; font-weight:bold; padding:15px; text-align:center; border-radius:8px;"

        judgment = "불량품" if proba >= 0.2 else "양품"

        return ui.div(
            [
                ui.HTML(f"예상 불량률: {proba*100:.2f}%"),
                ui.br(),
                ui.HTML(f"최종 판정: <span style='font-size:22px;'>{judgment}</span>")
            ],
            style=style
        )

    @render.plot
    def feature_importance_plot():
        try:
            importances = model.named_steps["model"].feature_importances_
            feat_names = model.named_steps["preprocessor"].get_feature_names_out()
            imp_df = pd.DataFrame({"feature": feat_names, "importance": importances})
            imp_df = imp_df.sort_values("importance", ascending=False).head(10)

            plt.figure(figsize=(8,5))
            plt.barh(imp_df["feature"], imp_df["importance"])
            plt.gca().invert_yaxis()
            plt.title("변수 중요도 Top 10")
            plt.tight_layout()
        except Exception:
            plt.figure()
            plt.text(0.5,0.5,"변수 중요도 계산 불가",ha="center",va="center")

    @render.plot
    def distribution_plot():
        try:
            plt.figure(figsize=(8,5))
            df_good = df_predict[df_predict["passorfail"]==0]["biscuit_thickness"]
            df_bad = df_predict[df_predict["passorfail"]==1]["biscuit_thickness"]

            plt.hist(df_good, bins=30, alpha=0.6, label="양품")
            plt.hist(df_bad, bins=30, alpha=0.6, label="불량품")

            plt.axvline(df_predict["biscuit_thickness"].mean(), color="red", linestyle="--", label="평균")
            plt.legend()
            plt.title("비스킷 두께 분포 (양품 vs 불량)")
            plt.tight_layout()
        except Exception:
            plt.figure()
            plt.text(0.5,0.5,"분포 그래프 생성 불가",ha="center",va="center")

    @render.plot
    def process_trend_plot():
        try:
            mold_trend = df_predict.groupby("mold_code")["passorfail"].mean().sort_values(ascending=False)
            plt.figure(figsize=(8,5))
            mold_trend.plot(kind="bar")
            plt.ylabel("불량률")
            plt.title("금형 코드별 불량률")
            plt.tight_layout()
        except Exception:
            plt.figure()
            plt.text(0.5,0.5,"공정별 그래프 생성 불가",ha="center",va="center")

    # # ===== 데이터 요약 카드 =====
    # @output
    # @render.table
    # def df_summary():
    #     return pd.DataFrame({
    #         "항목": ["행 개수", "열 개수", "총 결측치", "결측치 비율(%)",
    #             "수치형 변수 개수", "범주형 변수 개수"],
    #         "값": [
    #             f"{df_raw.shape[0]:,}",
    #             f"{df_raw.shape[1]:,}",
    #             f"{df_raw.isna().sum().sum():,}",
    #             round(df_raw.isna().sum().sum() / (df_raw.shape[0]*df_raw.shape[1]) * 100, 2),
    #             df_raw.select_dtypes(include="number").shape[1],
    #             df_raw.select_dtypes(exclude="number").shape[1],
    #         ]
    #     })

    # # --- 결측치 비율 ---
    # @output
    # @render.plot
    # def missing_plot():
    #     # 결측치 비율 계산
    #     na_ratio = (df_explore.isna().mean() * 100)
    #     na_ratio = na_ratio[na_ratio > 0].sort_values(ascending=False).head(6)  # 상위 6개만

    #     if na_ratio.empty:
    #         fig, ax = plt.subplots()
    #         ax.text(0.5, 0.5, "결측치가 있는 컬럼이 없습니다.", ha="center", va="center")
    #         ax.axis("off")
    #         return fig

    #     fig, ax = plt.subplots(figsize=(6, 3))
    #     sns.barplot(x=na_ratio.index, y=na_ratio.values, ax=ax, color="tomato")

    #     # 막대 위에 라벨 표시
    #     for i, v in enumerate(na_ratio.values):
    #         ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    #     ax.set_ylabel("결측치 비율 (%)")
    #     ax.set_xlabel("컬럼명")
    #     ax.set_title("결측치 비율 상위 6개 컬럼")
    #     ax.tick_params(axis="x", rotation=45)
    #     ax.set_ylim(0, max(na_ratio.values) * 1.2)  # 여백 확보

    #     return fig

    # --- 변수 타입 분포 ---
    # @output
    # @render.plot
    # def dtype_pie():
    #     num = df_explore.select_dtypes(include="number").shape[1]
    #     cat = df_explore.select_dtypes(exclude="number").shape[1]
    #     fig, ax = plt.subplots()
    #     ax.pie([num, cat], labels=["수치형", "범주형"], autopct="%1.1f%%", colors=["skyblue", "orange"])
    #     ax.set_title("변수 타입 비율")
    #     return fig

    # # --- 수치형 변수 상관 관계 히트맵 ---
    # @output
    # @render.plot
    # def corr_heatmap_overview():
    #     num_df = df_explore.select_dtypes(include="number")
    #     if num_df.shape[1] < 2:
    #         fig, ax = plt.subplots()
    #         ax.text(0.5, 0.5, "수치형 변수가 부족합니다.", ha="center", va="center")
    #         ax.axis("off")
    #         return fig

    #     corr = num_df.corr(numeric_only=True)
    #     fig, ax = plt.subplots(figsize=(5, 4))
    #     sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, cbar=True)
    #     ax.set_title("수치형 변수 상관관계 히트맵")
    #     return fig

    # --- 동적 필터 UI ---
    @output
    @render.ui
    def filter_ui():
        var = input.var()
        if var not in df_explore.columns:
            return None

        # registration_time → datetime slider (10분 단위)
        if var == "registration_time":
            times = pd.to_datetime(df_explore["registration_time"], errors="coerce")
            times = times.dropna()
            if times.empty:
                return ui.markdown("⚠️ registration_time 컬럼에 유효한 datetime 값이 없습니다.")
            min_t, max_t = times.min(), times.max()

            # 초기 범위: 최대값 - 10분 ~ 최대값
            min_t, max_t = times.min(), times.max()
            # init_end = min_t + pd.Timedelta(minutes=10)
            # if init_end > max_t:
            #     init_end = max_t

            return ui.input_slider(
                "ts_range",
                "시간 범위 선택",
                min=min_t, max=max_t,
                value=[min_t, max_t],
                step=600,
                time_format="%Y-%m-%d %H:%M"
            )

        # 범주형 변수
        if not pd.api.types.is_numeric_dtype(df_explore[var]):
            categories = df_explore[var].dropna().astype(str).unique().tolist()
            categories = sorted(categories) + ["없음"]
            return ui.input_checkbox_group(
                "filter_val",
                f"{label_map.get(var, var)} 선택",
                choices=categories,
                selected=categories
            )

        # 수치형 변수
        min_val, max_val = df_explore[var].min(), df_explore[var].max()
        return ui.input_slider(
            "filter_val",
            f"{label_map.get(var, var)} 범위",
            min=min_val, max=max_val,
            value=[min_val, max_val]
        )
    
    # --- 데이터 필터링 ---
    @reactive.calc
    def filtered_df():
        dff = df_explore.copy()
        var = input.var()

        if var in dff.columns and "filter_val" in input:
            rng = input.filter_val()
            if rng is None:
                return dff

            # registration_time 필터
            if var == "registration_time":
                dff["registration_time"] = pd.to_datetime(dff["registration_time"], errors="coerce")
                dff = dff.dropna(subset=["registration_time"])
                start, end = pd.to_datetime(rng[0]), pd.to_datetime(rng[1])
                dff = dff[(dff["registration_time"] >= start) & (dff["registration_time"] <= end)]

            # 범주형 필터
            elif not pd.api.types.is_numeric_dtype(dff[var]):
                selected = rng
                if "없음" in selected:
                    dff = dff[(dff[var].isin([x for x in selected if x != "없음"])) | (dff[var].isna()) | (dff[var]=="")]
                else:
                    dff = dff[dff[var].isin(selected)]

            # 수치형 필터
            else:
                start, end = float(rng[0]), float(rng[1])
                dff = dff[(dff[var] >= start) & (dff[var] <= end)]

        return dff

    @output
    @render.plot
    def local_factor_plot():
     factors = local_factors()
     if factors is None or factors.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "아직 예측을 실행하지 않았습니다.", ha="center", va="center")
        ax.axis("off")
        return fig

     top = factors.head(5).copy()
     top["importance"] = top["importance"] * 100  # % 변환

     fig, ax = plt.subplots(figsize=(8, 4))
     bars = ax.barh(top["feature"], top["importance"], color="tomato")

    # 각 막대 끝에 % 숫자 표시
     for bar, val in zip(bars, top["importance"]):
        ax.text(bar.get_width() + 0.5,
                bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%",
                va="center")

        ax.invert_yaxis()
        ax.set_xlabel("기여도 (%)")
        ax.set_title("이번 케이스 불량 기여 요인 Top 5")
        plt.tight_layout()
     return fig

    # === 여기에 local_factor_desc() 붙여넣기 ===
    @output
    @render.ui
    def local_factor_desc():
     factors = local_factors()
     if factors is None or factors.empty:
        return ui.markdown("아직 예측을 실행하지 않았습니다.")

     top = factors.head(5).copy()
     top["importance"] = top["importance"] * 100

     exclude_vars = ["count", "monthly_count", "global_count"]
     use_num_cols = [c for c in num_cols if c not in exclude_vars]
     baseline = df_predict[df_predict["passorfail"] == 0][use_num_cols].mean()
     current = get_input_data().iloc[0][use_num_cols]

     rows_html = []
     for _, row in top.iterrows():
        feat = row["feature"]
        importance = row["importance"]

        col = [k for k, v in label_map.items() if v == feat]
        if not col: 
            continue
        col = col[0]

        left_text = f"{feat}: {importance:.1f}%"

        if col in current.index:
            diff = current[col] - baseline[col]
            if abs(diff) > 1e-6:
                direction = "낮추세요" if diff > 0 else "올리세요"
                adj_val = abs(diff) / 2
                right_text = f"{adj_val:.1f} 단위 {direction} (현재 {current[col]:.1f}, 기준 {baseline[col]:.1f})"
            else:
                right_text = "-"
        else:
            right_text = "-"

        row_html = f"""
        <div style='display:flex; align-items:center; margin-bottom:8px; font-size:15px;'>
            <div style='flex:1; text-align:left;'>{left_text}</div>
            <div style='flex:0.2; text-align:center;'>
                <i class="fa-solid fa-arrow-right fa-beat" style="color:#007bff;"></i>
            </div>
            <div style='flex:2; text-align:left; color:#444;'>{right_text}</div>
        </div>
        """
        rows_html.append(row_html)

    # 🔹 for문 끝난 뒤에 return 실행
     return ui.div(
        [
            ui.markdown("**이번 예측에서 불량률은 아래 요인들의 영향을 많이 받습니다:**"),
            ui.HTML("".join(rows_html)),
            ui.input_action_button(
                "apply_suggestions", "반영하고 다시 예측하기",
                class_="btn btn-warning", style="margin-top:15px;"
            )
        ]
    )
     
    @output
    @render.ui
    def ts_filter_ui():
        if "registration_time" not in df_raw.columns:
            return ui.markdown("⚠️ registration_time 없음")

        times = pd.to_datetime(df_raw["registration_time"], errors="coerce").dropna()
        if times.empty:
            return ui.markdown("⚠️ 유효한 datetime 값 없음")

        min_t, max_t = times.min().date(), times.max().date()

        # 🔽 기존 input_date_range 대신 → input_date 두 개
        return ui.div(
            ui.input_date(
                "ts_start", "from",
                value=min_t, min=min_t, max=max_t,
                width="200px"
            ),
            ui.input_date(
                "ts_end", "to",
                value=max_t, min=min_t, max=max_t,
                width="200px"
            ),
            style="display:flex; flex-direction:column; gap:8px;"  # 두 줄 배치
        )

    @output
    @render.plot
    def dist_plot():
        try:
            var = input.var()
            mold = input.mold_code2()
            dff = df_explore[df_explore["mold_code"].astype(str) == mold]

            if var not in dff.columns:
                fig, ax = plt.subplots()
                ax.text(0.5,0.5,"선택한 변수가 데이터에 없음",ha="center",va="center")
                ax.axis("off")
                return fig

            fig, ax = plt.subplots(figsize=(6,4))
            if pd.api.types.is_numeric_dtype(dff[var]):
                sns.histplot(dff[var], bins=30, kde=True, ax=ax, color="tomato")
            else:
                dff[var].value_counts().plot(kind="bar", ax=ax, color="tomato")

            ax.set_title(f"{get_label(var)} 분포 (Mold {mold})")
            return fig

        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5,0.5,f"에러: {e}",ha="center",va="center")
            ax.axis("off")
            return fig

    # Boxplot 원본 선택 시 → 파생 자동 없음
    @reactive.Effect
    @reactive.event(input.box_var)
    def _():
        if input.box_var() != "없음":
            update_select("box_var_derived", selected="없음")

    # Boxplot 파생 선택 시 → 원본 자동 없음
    @reactive.Effect
    @reactive.event(input.box_var_derived)
    def _():
        if input.box_var_derived() != "없음":
            update_select("box_var", selected="없음")

    @output
    @render_plotly
    def timeseries_plot():
        if "registration_time" not in df_raw.columns:
            return px.scatter(title="⚠️ registration_time 없음")

        # 변수 선택 처리
        var = None

        # 원본 선택된 경우
        if input.ts_var() != "없음":
            # 한글 라벨 → 컬럼명 변환
            inv_map = {v: k for k, v in label_map.items()}
            var = inv_map.get(input.ts_var(), input.ts_var())

        # 파생 선택된 경우 (이미 컬럼명 그대로라 역매핑 불필요)
        elif input.ts_var_derived() != "없음":
            derived_map = {
                "상/하부 주입 속도 비율": "speed_ratio",
                "주입 압력 비율": "pressure_speed_ratio",
            }
            var = derived_map.get(input.ts_var_derived(), input.ts_var_derived())

        # 아무 것도 선택 안 한 경우
        if var is None:
            return px.scatter(title="⚠️ 변수 선택 필요")
        
        rng_start = pd.to_datetime(input.ts_start())
        rng_end   = pd.to_datetime(input.ts_end())

        # dff = df_raw.copy()
        # ✅ 원본 + 파생 변수가 모두 있는 df_explore 사용
        dff = df_explore.copy()
        
        # df_explore에는 시간/라벨이 없으므로 df_raw에서 가져와 붙여줌
        dff["registration_time"] = pd.to_datetime(df_raw["registration_time"], errors="coerce")
        dff["passorfail"] = df_raw["passorfail"].values
        
        # 결측/범위 필터링
        dff = dff.dropna(subset=["registration_time", var, "passorfail"])
        dff = dff[(dff["registration_time"] >= rng_start) & (dff["registration_time"] <= rng_end)]

        if dff.empty:
            return px.scatter(title="⚠️ 선택한 구간에 데이터 없음")

        # Pass/Fail → 색상
        dff["불량여부"] = dff["passorfail"].map({0: "Pass", 1: "Fail"})
        dff = dff.sort_values("registration_time")
        dff["registration_time_str"] = dff["registration_time"].dt.strftime("%Y-%m-%d %H:%M:%S")

        # === 원본 점 그래프 ===
        fig = px.scatter(
            dff,
            x="registration_time_str",
            y=var,
            color="불량여부",
            color_discrete_map={"Pass": "green", "Fail": "red"},
            title=f"{label_map.get(var, var)} 시계열 (원본{' + 스무딩' if pd.api.types.is_numeric_dtype(dff[var]) else ''})",
            labels={
                "registration_time_str": "등록 시간",
                var: label_map.get(var, var)
            },
        )

        # =========================
        # 📦 Boxplot 비교 (원본 vs 전처리)
        # =========================
        # 원본 Boxplot (Matplotlib)
        @output
        @render.plot
        def boxplot_raw():
            var = None
            if input.box_var() != "없음":
                inv_map = {v: k for k, v in label_map.items()}
                var = inv_map.get(input.box_var(), input.box_var()).lower()
            elif input.box_var_derived() != "없음":
                derived_map = {
                    "상/하부 주입 속도 비율": "speed_ratio",
                    "주입 압력 비율": "pressure_speed_ratio",
                }
                var = derived_map.get(input.box_var_derived(), input.box_var_derived())
        
            if var is None or var not in df_explore.columns:
                fig, ax = plt.subplots()
                ax.text(0.5,0.5,"⚠️ 변수 선택 필요",ha="center",va="center")
                ax.axis("off")
                return fig
        
            dff = df_explore[[var, "passorfail"]].dropna().copy()
            if dff.empty:
                fig, ax = plt.subplots()
                ax.text(0.5,0.5,"⚠️ 원본 데이터 없음",ha="center",va="center")
                ax.axis("off")
                return fig
        
            dff["불량여부"] = dff["passorfail"].map({0:"Pass",1:"Fail"})
        
            fig, ax = plt.subplots(figsize=(6,4))
            sns.boxplot(x="불량여부", y=var, data=dff, hue="불량여부", palette={"Pass":"green","Fail":"red"}, ax=ax)
            ax.set_title(f"{label_map.get(var, var)} - 원본 Boxplot")
            return fig
        
        
        # 전처리 Boxplot (Matplotlib)
        @output
        @render.plot
        def boxplot_proc():
            var = None
            if input.box_var() != "없음":
                inv_map = {v: k for k, v in label_map.items()}
                var = inv_map.get(input.box_var(), input.box_var()).lower()
            elif input.box_var_derived() != "없음":
                derived_map = {
                    "상/하부 주입 속도 비율": "speed_ratio",
                    "주입 압력 비율": "pressure_speed_ratio",
                }
                var = derived_map.get(input.box_var_derived(), input.box_var_derived())
        
            if var is None or var not in df_processed.columns:
                fig, ax = plt.subplots()
                ax.text(0.5,0.5,"⚠️ 변수 선택 필요",ha="center",va="center")
                ax.axis("off")
                return fig
        
            dff = df_processed[[var, "passorfail"]].dropna().copy()
            if dff.empty:
                fig, ax = plt.subplots()
                ax.text(0.5,0.5,"⚠️ 전처리 데이터 없음",ha="center",va="center")
                ax.axis("off")
                return fig
        
            dff["불량여부"] = dff["passorfail"].map({0:"Pass",1:"Fail"})
        
            fig, ax = plt.subplots(figsize=(6,4))
            sns.boxplot(x="불량여부", y=var, data=dff, hue="불량여부", palette={"Pass":"green","Fail":"red"}, ax=ax)
            ax.set_title(f"{label_map.get(var, var)} - 전처리 Boxplot")
            return fig



        # # === 수치형일 때만 스무딩 추가 ===
        if pd.api.types.is_numeric_dtype(dff[var]):
            def smooth_series(series, window=20):
                # 이전 시점 기준 rolling mean
                return series.rolling(window=window, center=False, min_periods=1).mean()

            dff["smoothed"] = smooth_series(dff[var], window=20)

            fig.add_scatter(
                x=dff["registration_time_str"],
                y=dff["smoothed"],
                mode="lines",
                name=f"{label_map.get(var, var)} (Smoothed)",
                line=dict(color="blue", width=2)
            )



        # 공통 레이아웃
        fig.update_layout(
            title=dict(
                text=f"{label_map.get(var, var)} 시계열 (원본{' + 스무딩' if pd.api.types.is_numeric_dtype(dff[var]) else ''})",
                x=0.5,
                xanchor="center",
                y=0.98,         # 🔹 제목 더 위로
                yanchor="top",
                font=dict(size=18)
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            hovermode="x unified",
            margin=dict(l=40, r=20, t=80, b=40),  # 🔹 위쪽 여백 확보
            legend_title_text="",
            xaxis=dict(showgrid=True, gridcolor="lightgray", griddash="dot"),
            yaxis=dict(showgrid=True, gridcolor="lightgray", griddash="dot")
        )

        fig.update_traces(marker=dict(size=5, opacity=0.5))

        fig.update_xaxes(
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1h", step="hour", stepmode="backward"),
                    dict(count=6, label="6h", step="hour", stepmode="backward"),
                    dict(count=12, label="12h", step="hour", stepmode="backward"),
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(step="all", label="전체")
                ])
            )
        )

        return fig
    
    #======================================================
    # 금형코드 생산계획 시뮬레이션
    #======================================================
    # 동적 입력박스
    @render.ui
    def mold_inputs():
        inputs = []
        for code in codes[:-1]:
            color = mold_colors[code]
            label_html = ui.HTML(f"<span style='color:{color}; font-weight:bold;'>금형코드 {code}</span>")
            inputs.append(
                ui.input_numeric(f"target_{code}", label_html, value=0, min=0, step=100)
            )
        return inputs

    # 남은 생산량 표시
    @render.text
    def remaining_qty():
        total_target = input.monthly_target()
        user_sum = sum(input[f"target_{code}"]() for code in codes[:-1])
        remaining = total_target - user_sum
        if user_sum > total_target:
            return f"⚠️ 총합 {user_sum:,}개가 월 생산량 목표 {total_target:,}개를 초과했습니다!"
        else:
            return f"남은 생산량 : {remaining:,}개 (금형코드 {last_code}에 할당)"




    # 3. 몰드코드 요약 (한글화 + 소수점 2자리)
    @render.data_frame
    def mold_summary_table():
        df = mold_summary.copy()
        df = df.rename(columns={
            "mold_code": "금형코드",
            "avg_facility_cycleTime": "평균설비사이클(초)",
            "daily_capacity": "일일생산능력(이론)",
            "min_prod": "최소일일생산량",
            "max_prod": "최대일일생산량",
            "avg_prod": "평균일일생산량"
        })
        # 수치형 포맷 적용 (소수점 둘째 자리까지)
        df = df.round(2)
        return df

    # 시뮬레이션 실행 (버튼 클릭 시에만)
    @reactive.event(input.run_plan)
    def plan_df():
        total_target = input.monthly_target()
        year, month = int(input.year()), int(input.month())

        targets = {}
        user_sum = 0
        for code in codes[:-1]:
            qty = input[f"target_{code}"]()
            targets[code] = qty
            user_sum += qty
        targets[last_code] = max(total_target - user_sum, 0)

        if sum(targets.values()) == 0:
            for _, row in mold_summary.iterrows():
                code = row["mold_code"]
                ratio = row["daily_capacity"] / mold_summary["daily_capacity"].sum()
                targets[code] = int(total_target * ratio)

        # === 해당 월의 실제 일수 반영 ===
        _, last_day = calendar.monthrange(year, month)

        weeks = ["3종류", "2종류", "3종류", "2종류"]
        codes_3, codes_2 = codes[:3], codes[3:5]

        schedule = []
        day_counter = 0
        for week_num, mode in enumerate(weeks, start=1):
            if day_counter >= last_day:
                break
            selected = codes_3 if mode == "3종류" else codes_2
            daily_sum = sum(
                mold_summary.loc[mold_summary["mold_code"] == c, "daily_capacity"].values[0]
                for c in selected
            )
            ratios = {
                c: mold_summary.loc[mold_summary["mold_code"] == c, "daily_capacity"].values[0] / daily_sum
                for c in selected
            }
            for day in range(1, 8):
                day_counter += 1
                if day_counter > last_day:
                    break
                for code in codes:
                    if code in selected:
                        total_target_code = targets[code]
                        daily_plan = int((total_target_code / last_day) * ratios[code] * len(selected))
                    else:
                        daily_plan = 0
                    schedule.append({
                        "date": datetime.date(year, month, day_counter),
                        "week": week_num,
                        "day": day,
                        "mold_code": code,
                        "plan_qty": daily_plan
                    })
        return pd.DataFrame(schedule)

    # 달력형 뷰 (버튼 클릭 시에만 갱신)
    @render.ui
    @reactive.event(input.run_plan)
    def calendar_view():
        df = plan_df()
        year, month = int(input.year()), int(input.month())
        calendar.setfirstweekday(calendar.SUNDAY)
        days_kr = ["일", "월", "화", "수", "목", "금", "토"]
        cal = calendar.monthcalendar(year, month)

        html = '<div style="display:grid; grid-template-columns: 80px repeat(7, 1fr); gap:4px;">'
        html += '<div></div>' + "".join([f"<div style='font-weight:bold; text-align:center;'>{d}</div>" for d in days_kr])

        for w_i, week in enumerate(cal, start=1):
            html += f"<div style='font-weight:bold;'>{w_i}주</div>"
            for d in week:
                if d == 0:
                    html += "<div style='border:1px solid #ccc; min-height:80px; background:#f9f9f9;'></div>"
                else:
                    cell_date = datetime.date(year, month, d)
                    cell_df = df[df["date"] == cell_date]

                    cell_html = ""
                    for _, r in cell_df.iterrows():
                        if r["plan_qty"] > 0:
                            code = str(r["mold_code"])

                            # 세팅값 조회
                            row = setting_df[setting_df["mold_code"] == code]
                            if row.empty:
                                tooltip_html = "<p>세팅값 없음</p>"
                            else:
                                settings = row.to_dict("records")[0]

                            # HTML 표 생성
                            rows_html = "".join([
                                f"<tr><td>{label_map.get(k, k)}</td><td>{f'{v:.2f}' if isinstance(v, (int, float)) else v}</td></tr>"
                                for k, v in settings.items() if k != "mold_code"
                            ])
                            tooltip_html = f"""
                            <table class="table table-sm table-bordered" style="font-size:11px; background:white; color:black;">
                                <thead><tr><th>변수</th><th>값</th></tr></thead>
                                <tbody>{rows_html}</tbody>
                            </table>
                            """

                            # 툴팁 적용
                            cell_html += str(
                                ui.tooltip(
                                    ui.span(
                                        f"{code}: {r['plan_qty']}",
                                        style=f"color:{mold_colors[code]}; font-weight:bold;"
                                    ),
                                    ui.HTML(tooltip_html),  # 표 형태 툴팁
                                    placement="right"
                                )
                            ) + "<br>"

                    html += f"<div style='border:1px solid #ccc; min-height:80px; padding:4px; font-size:12px;'>{d}<br>{cell_html}</div>"
        html += "</div>"
        return ui.HTML(html)
    
    # ================================
    # 생산 시뮬레이션 탭 비율 그래프
    # ================================
    @output
    @render.plot
    def mold_plot():
        fig, ax = plt.subplots(figsize=(12, 6))
        pivot_count.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title("날짜별 금형 코드 생산 개수")
        ax.set_xlabel("날짜")
        ax.set_ylabel("생산 개수")
        ax.legend(title="금형 코드")
        plt.tight_layout()
        return fig

    # 버튼 클릭 → 모달 띄우기
    @reactive.effect
    @reactive.event(input.show_modal)  # 버튼 클릭 감지
    def _():
        ui.modal_show(
            ui.modal(
                ui.output_plot("mold_plot"),
                title="날짜별 금형 코드 생산 추이",
                easy_close=True,
                size="xl",
                footer=ui.modal_button("닫기")
            )
        )

app = App(app_ui, server, static_assets=app_dir / "www")
