

















import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# ----------------------
# Custom Transformers
# ----------------------

class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """date, time → datetime, hour, shift, global_count, monthly_count 생성"""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        if "date" in df.columns and "time" in df.columns:
            df["datetime"] = pd.to_datetime(
                df["date"].astype(str) + " " + df["time"].astype(str),
                errors="coerce",
                infer_datetime_format=True
            )

        df["hour"] = df["datetime"].dt.hour
        df["shift"] = df["hour"].apply(lambda h: "Day" if 8 <= h <= 19 else "Night")

        # global_count
        global_counts, accum, prev_count = [], 0, df.loc[0, "count"]
        for current_count in df["count"]:
            if current_count < prev_count:
                accum += prev_count
            global_counts.append(accum + current_count)
            prev_count = current_count
        df["global_count"] = global_counts

        # monthly_count
        df["year_month"] = df["datetime"].dt.to_period("M")
        df["monthly_count"] = df.groupby("year_month").cumcount() + 1

        return df

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """파생변수 + 결측치 처리"""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # 파생변수
        df["speed_ratio"] = df["low_section_speed"] / df["high_section_speed"]
        df["pressure_speed_ratio"] = df["cast_pressure"] / df["high_section_speed"]

        # 예외 처리 (0 나눗셈 → -1)
        df.loc[(df["low_section_speed"] == 0) & (df["high_section_speed"] == 0), "speed_ratio"] = -1
        df.loc[(df["low_section_speed"] != 0) & (df["high_section_speed"] == 0), "speed_ratio"] = -1
        df.loc[df["high_section_speed"] == 0, "pressure_speed_ratio"] = -1

        # heating_furnace 기본값
        if "heating_furnace" in df.columns:
            df["heating_furnace"] = df["heating_furnace"].fillna("C")

        # molten_temp → 최빈값
        if "molten_temp" in df.columns:
            df["molten_temp"] = df["molten_temp"].fillna(df["molten_temp"].mode()[0])

        # molten_volume → 보간
        if "molten_volume" in df.columns:
            df["molten_volume"] = df["molten_volume"].interpolate(method="linear").ffill().bfill()

        # ✅ 최종 안전장치: 모든 inf/-inf → -1
        df = df.replace([np.inf, -np.inf], -1)

        return df




class DropColumns(BaseEstimator, TransformerMixin):
    """불필요한 컬럼 삭제"""
    def __init__(self, drop_cols=None):
        self.drop_cols = drop_cols if drop_cols else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        return df.drop(columns=[c for c in self.drop_cols if c in df.columns])


# ----------------------
# 파이프라인 정의
# ----------------------
def build_pipeline():
    drop_cols = [
        "id", "line", "name", "mold_name",
        "date", "time", "registration_time",
        "year_month", "hour", "datetime", "real_time",
        "working"
    ]

    multi_cols = ["mold_code", "heating_furnace", "EMS_operation_time"]
    binary_cols = ["emergency_stop", "tryshot_signal"]

    cat_encoder = ColumnTransformer(
        transformers=[
            ("multi", OneHotEncoder(handle_unknown="ignore", sparse_output=False), multi_cols),
            ("shift", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["shift"]),
            ("binary", OneHotEncoder(handle_unknown="ignore", sparse_output=False), binary_cols),
        ],
        remainder="passthrough"
    )

    pipeline = Pipeline(steps=[
        ("datetime", DatetimeFeatureExtractor()),
        ("engineer", FeatureEngineer()),
        ("drop", DropColumns(drop_cols=drop_cols)),
        ("encode", cat_encoder)
    ])
    return pipeline




# ----------------------
# DataFrame 변환 함수
# ----------------------

def pipeline_to_dataframe(pipeline, X, fit=False):
    """파이프라인 실행 후 DataFrame 반환"""
    if fit:
        Xt = pipeline.fit_transform(X)
    else:
        Xt = pipeline.transform(X)

    feature_names = []
    enc = pipeline.named_steps["encode"]

    for name, trans, cols in enc.transformers_:
        if name in ["multi", "shift", "binary"]:   # ✅ binary 추가
            feature_names.extend(trans.get_feature_names_out(cols))
        elif name == "remainder":
            feature_names.extend(cols)

    return pd.DataFrame(Xt, columns=feature_names, index=X.index)


# ----------------------
# 실행
# ----------------------
if __name__ == "__main__":
    train = pd.read_csv("./data/train.csv")
    test = pd.read_csv("./data/test.csv")

    y_train = train["passorfail"]
    X_train = train.drop(columns=["passorfail"])
    X_test = test.copy()

    pipeline = build_pipeline()

    # train 변환 (fit + transform → DataFrame 반환)
    X_train_processed = pipeline_to_dataframe(pipeline, X_train, fit=True)

    # test 변환 (transform → DataFrame 반환)
    X_test_processed = pipeline_to_dataframe(pipeline, X_test, fit=False)

    print("✅ Train/Test 전처리 완료")
    print("Train shape:", X_train_processed.shape)
    print("Test shape :", X_test_processed.shape)
    print("Train columns:", X_train_processed.columns[:20].tolist())

# ----------------------
# 모델 학습 (RandomForest + SMOTENC)
# ----------------------# ----------------------
# 모델 학습 (RandomForest + SMOTENC)
# ----------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
import pandas as pd
import numpy as np

# 1) 학습/검증 데이터 분리
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_processed, y_train,
    test_size=0.2, random_state=42, stratify=y_train
)

# 2) NaN, inf 처리 (학습/검증 둘 다 안전하게)
X_tr = X_tr.replace([np.inf, -np.inf], -1).fillna(-1)
X_val = X_val.replace([np.inf, -np.inf], -1).fillna(-1)

# 3) 범주형 컬럼 인덱스 찾기
categorical_cols = [c for c in X_train_processed.columns 
                    if c.startswith(("mold_code_", "heating_furnace_", "EMS_operation_time_", "shift_"))]
categorical_features = [X_train_processed.columns.get_loc(c) for c in categorical_cols]

# 4) SMOTENC 적용 (학습 데이터만)
smote_nc = SMOTENC(categorical_features=categorical_features, random_state=42)
X_tr_res, y_tr_res = smote_nc.fit_resample(X_tr, y_tr)

print("Before SMOTE :", y_tr.value_counts().to_dict())
print("After SMOTE  :", pd.Series(y_tr_res).value_counts().to_dict())

# 5) RandomForest 모델 정의
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

# 6) 학습
rf.fit(X_tr_res, y_tr_res)

# 7) 검증 평가
y_pred = rf.predict(X_val)
print("✅ Validation Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))

# 8) 최종 test 예측 (SMOTE는 test에 적용 ❌)
X_test_processed = X_test_processed.replace([np.inf, -np.inf], -1).fillna(-1)
test_preds = rf.predict(X_test_processed)

# 9) 결과 저장
submission = pd.DataFrame({
    "id": test["id"],
    "prediction": test_preds
})
submission.to_csv("submission_rf.csv", index=False)
print("📁 submission_rf.csv 저장 완료")
# ----------------------
# 모델 학습 (RandomForest + SMOTENC + threshold 조정)
# ----------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
import matplotlib.pyplot as plt

# 1) Train/Validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_processed, y_train,
    test_size=0.2, random_state=42, stratify=y_train
)

# 2) NaN, inf 처리
X_tr = X_tr.replace([np.inf, -np.inf], -1).fillna(-1)
X_val = X_val.replace([np.inf, -np.inf], -1).fillna(-1)

# 3) 범주형 feature 인덱스
categorical_cols = [c for c in X_train_processed.columns 
                    if c.startswith(("mold_code_", "heating_furnace_", "EMS_operation_time_", "shift_",
                                     "emergency_stop_", "tryshot_signal_"))]
categorical_features = [X_train_processed.columns.get_loc(c) for c in categorical_cols]

# 4) SMOTENC
smote_nc = SMOTENC(categorical_features=categorical_features, random_state=42)
X_tr_res, y_tr_res = smote_nc.fit_resample(X_tr, y_tr)

print("Before SMOTE :", y_tr.value_counts().to_dict())
print("After SMOTE  :", pd.Series(y_tr_res).value_counts().to_dict())

# 5) RandomForest 정의 (불량 class에 가중치 강화 → Recall ↑)
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
    class_weight={0:1, 1:5}   # 불량(1)에 가중치 부여
)

# 6) 학습
rf.fit(X_tr_res, y_tr_res)

# 7) Validation 평가 (threshold 조정)
y_proba = rf.predict_proba(X_val)[:, 1]

# 기본 threshold=0.5
y_pred_default = (y_proba >= 0.5).astype(int)
print("=== Default threshold (0.5) ===")
print(classification_report(y_val, y_pred_default))

# 낮춘 threshold=0.3 → Recall ↑
threshold = 0.3
y_pred_custom = (y_proba >= threshold).astype(int)
print(f"=== Custom threshold ({threshold}) ===")
print(classification_report(y_val, y_pred_custom))

# 8) Precision-Recall Curve 시각화
precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)
plt.plot(thresholds, precisions[:-1], label="Precision")
plt.plot(thresholds, recalls[:-1], label="Recall")
plt.axvline(x=0.5, color="gray", linestyle="--", label="0.5 threshold")
plt.axvline(x=threshold, color="red", linestyle="--", label=f"{threshold} threshold")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.title("Precision-Recall vs Threshold")
plt.show()

# 9) 최종 Test 예측 (threshold 적용)
X_test_processed = X_test_processed.replace([np.inf, -np.inf], -1).fillna(-1)
test_proba = rf.predict_proba(X_test_processed)[:, 1]
test_preds = (test_proba >= threshold).astype(int)

# 10) 저장
submission = pd.DataFrame({
    "id": test["id"],
    "prediction": test_preds
})
submission.to_csv("submission_rf.csv", index=False)
print("📁 submission_rf.csv 저장 완료")


X_train



# 수치형 변수만 추출
X_train_numeric = X_train_processed.select_dtypes(include=['int64', 'float64'])

print(X_train_numeric.shape)   # 행/열 크기 확인
print(X_train_numeric.columns) # 어떤 컬럼이 뽑혔는지 확인

import pandas as pd
import numpy as np

# 수치형 데이터셋 (X_train_numeric 가정)
X_train_numeric = X_train.select_dtypes(include=[np.number])

results = {}

for col in X_train_numeric.columns:
    data = X_train_numeric[col].dropna()
    
    # IQR 계산
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    # 정상 범위 안 값만 사용
    filtered = data[(data >= lower_bound) & (data <= upper_bound)]
    mean = filtered.mean()
    std = filtered.std()
    
    # 각 값 σ 구간 라벨링
    def sigma_label(x):
        if x < mean - 4*std or x > mean + 4*std:
            return "outlier"
        for i in range(1, 5):
            if mean - i*std <= x <= mean + i*std:
                return f"within {i}σ"
        return "beyond 4σ"
    
    labels = data.apply(sigma_label)
    
    results[col] = {
        "mean": mean,
        "std": std,
        "lower_bound(IQR*3)": lower_bound,
        "upper_bound(IQR*3)": upper_bound,
        "σ 구간 분포": labels.value_counts().to_dict()
    }

# 요약 DataFrame
results_df = pd.DataFrame(results).T
print(results_df.head())   # 앞부분만 확인
