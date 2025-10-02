import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score, average_precision_score, precision_recall_curve

from imblearn.over_sampling import SMOTENC, SMOTE
import matplotlib.pyplot as plt


# ======================
# 1) 커스텀 전처리
# ======================
class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        df = X.copy()
        if "date" in df.columns and "time" in df.columns:
            df["datetime"] = pd.to_datetime(df["date"].astype(str)+" "+df["time"].astype(str), errors="coerce")
        if "datetime" in df.columns:
            df["hour"] = df["datetime"].dt.hour
            df["shift"] = df["hour"].apply(lambda h: "Day" if 8<=h<=19 else "Night")
            df["year_month"] = df["datetime"].dt.to_period("M")
            df["monthly_count"] = df.groupby("year_month").cumcount()+1
        return df

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X,y=None): return self
    def transform(self, X):
        df = X.copy()
        df["speed_ratio"] = df["low_section_speed"] / df["high_section_speed"]
        df["pressure_speed_ratio"] = df["cast_pressure"] / df["high_section_speed"]
        df.loc[df["high_section_speed"]==0, ["speed_ratio","pressure_speed_ratio"]] = -1
        return df

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, drop_cols=None): self.drop_cols = drop_cols or []
    def fit(self, X,y=None): return self
    def transform(self, X): return X.drop(columns=[c for c in self.drop_cols if c in X.columns])


# ======================
# 2) 유틸
# ======================
def find_best_threshold_fbeta(y_true, y_prob, beta=2.0):
    p, r, t = precision_recall_curve(y_true, y_prob)
    t = np.append(t, 1.0)
    fbeta = (1+beta**2)*(p*r)/(beta**2*p+r+1e-12)
    best_idx = int(np.nanargmax(fbeta))
    return float(t[best_idx]), float(fbeta[best_idx])


# ======================
# 3) 메인 실행
# ======================
if __name__ == "__main__":
    # ----- 데이터 로드
    train = pd.read_csv("./data/train.csv")
    test = pd.read_csv("./data/test.csv")

    y_train = train["passorfail"]
    X_train = train.drop(columns=["passorfail"])

    drop_cols_stage1 = ["id","line","name","mold_name","registration_time","real_time","working"]
    drop_cols_stage2 = ["datetime","hour","year_month"]

    base_preproc = Pipeline(steps=[
        ("datetime", DatetimeFeatureExtractor()),
        ("engineer", FeatureEngineer()),
        ("drop1", DropColumns(drop_cols=drop_cols_stage1)),
        ("drop2", DropColumns(drop_cols=drop_cols_stage2)),
    ])

    tmp_after = base_preproc.fit_transform(X_train)
    expected_cats = ["mold_code","heating_furnace","EMS_operation_time","shift","emergency_stop","tryshot_signal"]
    present_cats = [c for c in expected_cats if c in tmp_after.columns]

    ord_cat_pipe = Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                             ("ord",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1))])
    num_imp_pipe = Pipeline([("imp",SimpleImputer(strategy="median"))])
    num_selector = make_column_selector(dtype_include=np.number)

    prep_for_smote = ColumnTransformer([
        ("cat_ord", ord_cat_pipe, present_cats),
        ("num_imp", num_imp_pipe, num_selector),
    ], remainder="drop")

    X_prepped = prep_for_smote.fit_transform(base_preproc.fit_transform(X_train), y_train)

    # --- SMOTE/SMOTENC ---
    if len(present_cats) > 0:
        sampler = SMOTENC(categorical_features=list(range(len(present_cats))), random_state=42)
    else:
        sampler = SMOTE(random_state=42)
    X_res, y_res = sampler.fit_resample(X_prepped, y_train)

    # ----- 재인코딩
    if len(present_cats) > 0:
        post_reencode = ColumnTransformer([
            ("cat_ohe", OneHotEncoder(handle_unknown="ignore"), list(range(len(present_cats)))),
            ("num_std", StandardScaler(with_mean=False), slice(len(present_cats), None)),
        ])
    else:
        post_reencode = ColumnTransformer([
            ("num_std", StandardScaler(with_mean=False), slice(0, None)),
        ])

    # ----- 파라미터 그리드
    param_grid = {
        "logit__penalty": ["l2","elasticnet"],
        "logit__C": [0.01,0.05,0.1,0.2,0.5,1.0,2.0],
        "logit__l1_ratio": [0.0,0.2,0.5,0.8],
        "logit__class_weight": [None,"balanced"]
    }
    all_params = list(ParameterGrid(param_grid))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ----- 진행률 표시
    total_tasks = len(all_params)*cv.get_n_splits()
    results = []

    with tqdm(total=total_tasks, desc="GridSearch Progress", ncols=100) as pbar:
        for params in all_params:
            fold_scores = []
            for tr_idx, val_idx in cv.split(X_res, y_res):
                X_tr, X_val = X_res[tr_idx], X_res[val_idx]
                y_tr, y_val = y_res[tr_idx], y_res[val_idx]

                pipe = Pipeline([
                    ("post", post_reencode),
                    ("logit", LogisticRegression(
                        solver="saga", max_iter=5000, n_jobs=-1, random_state=42, **{
                            k.replace("logit__",""): v for k,v in params.items()
                        }
                    ))
                ])

                pipe.fit(X_tr, y_tr)
                y_pred = pipe.predict(X_val)
                y_prob = pipe.predict_proba(X_val)[:,1]

                fold_scores.append({
                    "f1": f1_score(y_val,y_pred),
                    "recall": recall_score(y_val,y_pred),
                    "roc_auc": roc_auc_score(y_val,y_prob),
                    "ap": average_precision_score(y_val,y_prob)
                })

                pbar.update(1)

            mean_scores = {m: np.mean([fs[m] for fs in fold_scores]) for m in fold_scores[0].keys()}
            results.append({**params, **mean_scores})

    # ----- 결과 DataFrame 저장
    cvres = pd.DataFrame(results)
    cvres.to_csv("logit_smote_tqdm_results.csv", index=False)
    print("\n📁 저장 완료: logit_smote_tqdm_results.csv")

    # ----- Top5 by F1 / Recall
    print("\n===== Top 5 (by F1) =====")
    print(cvres.sort_values("f1", ascending=False).head(5))

    print("\n===== Top 5 (by Recall) =====")
    print(cvres.sort_values("recall", ascending=False).head(5))
