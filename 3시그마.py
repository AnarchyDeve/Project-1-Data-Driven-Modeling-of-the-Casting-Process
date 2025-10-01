##############################################################
# Recall 기준
##########################################################
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import time
import optuna
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC
from xgboost import XGBClassifier


# ======================
# 1) 커스텀 전처리
# ======================
class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
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
        if "datetime" in df.columns:
            df["hour"] = df["datetime"].dt.hour
            df["shift"] = df["hour"].apply(lambda h: "Day" if 8 <= h <= 19 else "Night")
            prev_count = df["count"].iloc[0]
            global_counts, accum = [], 0
            for current_count in df["count"]:
                if current_count < prev_count:
                    accum += prev_count
                global_counts.append(accum + current_count)
                prev_count = current_count
            df["global_count"] = global_counts
            df["year_month"] = df["datetime"].dt.to_period("M")
            df["monthly_count"] = df.groupby("year_month").cumcount() + 1
        return df

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): 
        return self
    def transform(self, X):
        df = X.copy()
        if "low_section_speed" in df.columns and "high_section_speed" in df.columns:
            df["speed_ratio"] = df["low_section_speed"] / df["high_section_speed"]
            df["pressure_speed_ratio"] = df["cast_pressure"] / df["high_section_speed"]
            df.loc[(df["low_section_speed"] == 0) & (df["high_section_speed"] == 0), "speed_ratio"] = -1
            df.loc[(df["low_section_speed"] != 0) & (df["high_section_speed"] == 0), "speed_ratio"] = -1
            df.loc[df["high_section_speed"] == 0, "pressure_speed_ratio"] = -1

        for col in ["heating_furnace", "emergency_stop", "tryshot_signal", "EMS_operation_time"]:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")

        if "molten_temp" in df.columns and df["molten_temp"].isna().any():
            df["molten_temp"] = df["molten_temp"].fillna(df["molten_temp"].mode()[0])

        if "molten_volume" in df.columns:
            df["molten_volume"] = df["molten_volume"].interpolate("linear").ffill().bfill()

        df = df.replace([np.inf, -np.inf], -1)
        return df

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, drop_cols=None):
        self.drop_cols = drop_cols or []
    def fit(self, X, y=None): 
        return self
    def transform(self, X):
        return X.drop(columns=[c for c in self.drop_cols if c in X.columns])


# ======================
# 2) Threshold Finder
# ======================
def find_best_threshold_fbeta(y_true, y_prob, beta=2.0):
    p, r, t = precision_recall_curve(y_true, y_prob)
    t = np.append(t, 1.0)
    fbeta = (1 + beta**2) * (p * r) / (beta**2 * p + r + 1e-12)
    best_idx = int(np.nanargmax(fbeta))
    return float(t[best_idx]), float(fbeta[best_idx])


# ======================
# 3) Main
# ======================
if __name__ == "__main__":
    # ----- 데이터 로드
    train = pd.read_csv("./data/train.csv")
    test  = pd.read_csv("./data/test.csv")

    y_train = train["passorfail"]
    X_train = train.drop(columns=["passorfail"])
    X_test  = test.copy()

    # ----- 드랍할 컬럼
    drop_cols = ["id","line","name","mold_name","date","time","registration_time",
                 "year_month","hour","datetime","real_time","working"]

    # ----- 카테고리/수치형 구분
    tmp_after = (DatetimeFeatureExtractor().fit_transform(X_train))
    tmp_after = (FeatureEngineer().fit_transform(tmp_after))
    tmp_after = DropColumns(drop_cols=drop_cols).fit_transform(tmp_after)

    expected_cats = ["mold_code","heating_furnace","EMS_operation_time","shift",
                     "emergency_stop","tryshot_signal"]
    present_cats = [c for c in expected_cats if c in tmp_after.columns]

    cat_pipe = SkPipeline(steps=[
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    num_pipe = SkPipeline(steps=[("imp", SimpleImputer(strategy="median"))])
    num_selector = make_column_selector(dtype_include=np.number)

    model_preproc = ColumnTransformer(
        transformers=[("cat", cat_pipe, present_cats),
                      ("num", num_pipe, num_selector)],
        remainder="drop"
    )

    categorical_feature_indices = list(range(len(present_cats)))

    # ----- 최종 파이프라인
    pipe = ImbPipeline(steps=[
        ("datetime", DatetimeFeatureExtractor()),
        ("engineer", FeatureEngineer()),
        ("drop", DropColumns(drop_cols=drop_cols)),
        ("prep", model_preproc),   # 결측치 처리 포함
        ("smote", SMOTENC(categorical_features=categorical_feature_indices, random_state=42)),
        ("xgb", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
            tree_method="hist"
        ))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ======================
    # Optuna Objective
    # ======================
    def objective(trial):
        params = {
            "smote__sampling_strategy": trial.suggest_float("smote__sampling_strategy", 0.1, 1.0),
            "smote__k_neighbors": trial.suggest_int("smote__k_neighbors", 3, 7),
            "xgb__n_estimators": trial.suggest_int("xgb__n_estimators", 300, 800),
            "xgb__learning_rate": trial.suggest_float("xgb__learning_rate", 0.01, 0.2, log=True),
            "xgb__max_depth": trial.suggest_int("xgb__max_depth", 3, 8),
            "xgb__min_child_weight": trial.suggest_int("xgb__min_child_weight", 1, 5),
            "xgb__subsample": trial.suggest_float("xgb__subsample", 0.5, 1.0),
            "xgb__colsample_bytree": trial.suggest_float("xgb__colsample_bytree", 0.5, 1.0),
            "xgb__gamma": trial.suggest_float("xgb__gamma", 0.0, 0.5),
            "xgb__reg_lambda": trial.suggest_float("xgb__reg_lambda", 0.5, 5.0, log=True),
            "xgb__reg_alpha": trial.suggest_float("xgb__reg_alpha", 0.0, 1.0),
            "xgb__scale_pos_weight": trial.suggest_float("xgb__scale_pos_weight", 1.0, 5.0),
        }
        pipe.set_params(**params)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="recall", n_jobs=-1)
        return scores.mean()

    # ======================
    # Optuna 실행
    # ======================
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, show_progress_bar=True)  # trial 수 조절 가능

    # ======================
    # 결과 출력
    # ======================
    print("\n===== Optuna 결과 =====")
    print("Best Params:", study.best_trial.params)
    print("Best Recall:", study.best_value)

    print("\n===== Top 10 Trials (Recall 기준) =====")
    top_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -1, reverse=True)[:10]
    for i, t in enumerate(top_trials, 1):
        print(f"Rank {i} | Recall={t.value:.4f} | Params={t.params}")


from sklearn.model_selection import cross_val_score

# Top 10 trial들
top_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:10]

print("\n===== Top 10 Trials (Recall/F1/Acc) =====")
for rank, t in enumerate(top_trials, 1):
    params = t.params
    pipe.set_params(**params)
    
    recall = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="recall", n_jobs=-1).mean()
    f1     = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1).mean()
    acc    = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1).mean()
    
    print(f"Rank {rank} | Recall={recall:.4f} | F1={f1:.4f} | Acc={acc:.4f} | Params={params}")

#####################################################

# ===== Top 10 Trials (Recall 기준) =====
# Rank 1 | Recall=0.9665 | Params={'smote__sampling_strategy': 0.8720679052176163, 'smote__k_neighbors': 3, 'xgb__n_estimators': 536, 'xgb__learning_rate': 0.011309626065281651, 'xgb__max_depth': 6, 'xgb__min_child_weight': 5, 'xgb__subsample': 0.7817757557752542, 'xgb__colsample_bytree': 0.7610341369684775, 'xgb__gamma': 0.317690601991742, 'xgb__reg_lambda': 1.4536900190258994, 'xgb__reg_alpha': 0.30245181417777384, 'xgb__scale_pos_weight': 4.511464146520451}
# Rank 2 | Recall=0.9643 | Params={'smote__sampling_strategy': 0.7335966172521562, 'smote__k_neighbors': 3, 'xgb__n_estimators': 467, 'xgb__learning_rate': 0.020633047621508027, 'xgb__max_depth': 5, 'xgb__min_child_weight': 5, 'xgb__subsample': 0.5004901203245256, 'xgb__colsample_bytree': 0.650126323739448, 'xgb__gamma': 0.17305910402530378, 'xgb__reg_lambda': 3.8722801203326043, 'xgb__reg_alpha': 0.3941824431159283, 'xgb__scale_pos_weight': 4.996686121900996}
# Rank 3 | Recall=0.9643 | Params={'smote__sampling_strategy': 0.9349318927976972, 'smote__k_neighbors': 3, 'xgb__n_estimators': 569, 'xgb__learning_rate': 0.035695536700790745, 'xgb__max_depth': 4, 'xgb__min_child_weight': 4, 'xgb__subsample': 0.6372443836612128, 'xgb__colsample_bytree': 0.6350485140990396, 'xgb__gamma': 0.19737583329506736, 'xgb__reg_lambda': 3.8331170255513443, 'xgb__reg_alpha': 0.5874044485123946, 'xgb__scale_pos_weight': 4.995667617332649}
# Rank 4 | Recall=0.9643 | Params={'smote__sampling_strategy': 0.7725164093513868, 'smote__k_neighbors': 3, 'xgb__n_estimators': 455, 'xgb__learning_rate': 0.02311736795959936, 'xgb__max_depth': 5, 'xgb__min_child_weight': 5, 'xgb__subsample': 0.8185197377844964, 'xgb__colsample_bytree': 0.6857768506727961, 'xgb__gamma': 0.21701344916614496, 'xgb__reg_lambda': 2.4283127317410798, 'xgb__reg_alpha': 0.06611769764790727, 'xgb__scale_pos_weight': 4.5217303229157935}
# Rank 5 | Recall=0.9643 | Params={'smote__sampling_strategy': 0.7512244909359673, 'smote__k_neighbors': 3, 'xgb__n_estimators': 468, 'xgb__learning_rate': 0.02645458478709883, 'xgb__max_depth': 5, 'xgb__min_child_weight': 5, 'xgb__subsample': 0.8933796097800385, 'xgb__colsample_bytree': 0.6546085569408211, 'xgb__gamma': 0.18897921948301327, 'xgb__reg_lambda': 3.9608189289676115, 'xgb__reg_alpha': 0.07562128645414914, 'xgb__scale_pos_weight': 4.683585487322524}
# Rank 6 | Recall=0.9640 | Params={'smote__sampling_strategy': 0.7596745040621485, 'smote__k_neighbors': 3, 'xgb__n_estimators': 467, 'xgb__learning_rate': 0.025484132975590136, 'xgb__max_depth': 5, 'xgb__min_child_weight': 5, 'xgb__subsample': 0.8604854416532041, 'xgb__colsample_bytree': 0.6601468936545319, 'xgb__gamma': 0.1801862854881622, 'xgb__reg_lambda': 3.938181975863292, 'xgb__reg_alpha': 0.009885698950640384, 'xgb__scale_pos_weight': 4.68106127387665}
# Rank 7 | Recall=0.9640 | Params={'smote__sampling_strategy': 0.8474342381800055, 'smote__k_neighbors': 3, 'xgb__n_estimators': 463, 'xgb__learning_rate': 0.035462461622069834, 'xgb__max_depth': 4, 'xgb__min_child_weight': 5, 'xgb__subsample': 0.8854560341897395, 'xgb__colsample_bytree': 0.6015262296521764, 'xgb__gamma': 0.1455472971790313, 'xgb__reg_lambda': 3.664230563556296, 'xgb__reg_alpha': 0.060573390008745565, 'xgb__scale_pos_weight': 4.818050147328701}
# Rank 8 | Recall=0.9637 | Params={'smote__sampling_strategy': 0.9669131610162475, 'smote__k_neighbors': 3, 'xgb__n_estimators': 521, 'xgb__learning_rate': 0.03402134900072943, 'xgb__max_depth': 4, 'xgb__min_child_weight': 3, 'xgb__subsample': 0.6343370273674797, 'xgb__colsample_bytree': 0.5799192204707221, 'xgb__gamma': 0.206562774816221, 'xgb__reg_lambda': 4.965013957087807, 'xgb__reg_alpha': 0.7342225126578525, 'xgb__scale_pos_weight': 4.33286884459314}
# Rank 9 | Recall=0.9634 | Params={'smote__sampling_strategy': 0.8220827068962212, 'smote__k_neighbors': 3, 'xgb__n_estimators': 560, 'xgb__learning_rate': 0.011723586227269335, 'xgb__max_depth': 4, 'xgb__min_child_weight': 5, 'xgb__subsample': 0.501653874581102, 'xgb__colsample_bytree': 0.6335633812156836, 'xgb__gamma': 0.4363129208826476, 'xgb__reg_lambda': 2.278162761299409, 'xgb__reg_alpha': 0.48057595554685933, 'xgb__scale_pos_weight': 4.805874169968876}
# Rank 10 | Recall=0.9634 | Params={'smote__sampling_strategy': 0.769486427420692, 'smote__k_neighbors': 3, 'xgb__n_estimators': 447, 'xgb__learning_rate': 0.02328154552808944, 'xgb__max_depth': 5, 'xgb__min_child_weight': 5, 'xgb__subsample': 0.8198855682650426, 'xgb__colsample_bytree': 0.6587533850290036, 'xgb__gamma': 0.2199559892151777, 'xgb__reg_lambda': 2.2120039847125437, 'xgb__reg_alpha': 0.10551744616035731, 'xgb__scale_pos_weight': 4.419781154392707}
##############################################################################
# f1  스코어 기준
##############################################################################
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import optuna
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTENC
from xgboost import XGBClassifier


# ======================
# 1) 커스텀 전처리
# ======================
class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
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
        prev_count = df["count"].iloc[0]
        global_counts, accum = [], 0
        for current_count in df["count"]:
            if current_count < prev_count:
                accum += prev_count
            global_counts.append(accum + current_count)
            prev_count = current_count
        df["global_count"] = global_counts
        df["year_month"] = df["datetime"].dt.to_period("M")
        df["monthly_count"] = df.groupby("year_month").cumcount() + 1
        return df

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): 
        return self
    def transform(self, X):
        df = X.copy()
        df["speed_ratio"] = df["low_section_speed"] / df["high_section_speed"]
        df["pressure_speed_ratio"] = df["cast_pressure"] / df["high_section_speed"]
        df.loc[(df["low_section_speed"] == 0) & (df["high_section_speed"] == 0), "speed_ratio"] = -1
        df.loc[(df["low_section_speed"] != 0) & (df["high_section_speed"] == 0), "speed_ratio"] = -1
        df.loc[df["high_section_speed"] == 0, "pressure_speed_ratio"] = -1
        for col in ["heating_furnace", "emergency_stop", "tryshot_signal", "EMS_operation_time"]:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")
        if "molten_temp" in df.columns and df["molten_temp"].isna().any():
            df["molten_temp"] = df["molten_temp"].fillna(df["molten_temp"].mode()[0])
        if "molten_volume" in df.columns:
            df["molten_volume"] = df["molten_volume"].interpolate("linear").ffill().bfill()
        df = df.replace([np.inf, -np.inf], -1)
        return df

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, drop_cols=None):
        self.drop_cols = drop_cols or []
    def fit(self, X, y=None): 
        return self
    def transform(self, X):
        return X.drop(columns=[c for c in self.drop_cols if c in X.columns])


# ======================
# 2) Threshold Finder
# ======================
def find_best_threshold_fbeta(y_true, y_prob, beta=2.0):
    p, r, t = precision_recall_curve(y_true, y_prob)
    t = np.append(t, 1.0)
    fbeta = (1 + beta**2) * (p * r) / (beta**2 * p + r + 1e-12)
    best_idx = int(np.nanargmax(fbeta))
    return float(t[best_idx]), float(fbeta[best_idx])


# ======================
# 3) Main
# ======================
if __name__ == "__main__":
    # ----- 데이터 로드
    train = pd.read_csv("./data/train.csv")
    test  = pd.read_csv("./data/test.csv")

    y_train = train["passorfail"]
    X_train = train.drop(columns=["passorfail"])
    X_test  = test.copy()

    # ----- 커스텀 전처리 후 컬럼 정의
    drop_cols = ["id","line","name","mold_name","date","time","registration_time",
                 "year_month","hour","datetime","real_time","working"]

    tmp_after = DatetimeFeatureExtractor().fit_transform(X_train)
    tmp_after = FeatureEngineer().fit_transform(tmp_after)
    tmp_after = DropColumns(drop_cols=drop_cols).fit_transform(tmp_after)

    expected_cats = ["mold_code","heating_furnace","EMS_operation_time","shift",
                     "emergency_stop","tryshot_signal"]
    present_cats = [c for c in expected_cats if c in tmp_after.columns]

    cat_pipe = SimpleImputer(strategy="most_frequent")
    num_pipe = SimpleImputer(strategy="median")
    num_selector = make_column_selector(dtype_include=np.number)

    model_preproc = ColumnTransformer(
        transformers=[("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), present_cats),
                      ("num", num_pipe, num_selector)],
        remainder="drop"
    )

    categorical_feature_indices = list(range(len(present_cats)))

    pipe = ImbPipeline(steps=[
        ("datetime", DatetimeFeatureExtractor()),
        ("engineer", FeatureEngineer()),
        ("drop", DropColumns(drop_cols=drop_cols)),
        ("prep", model_preproc),
        ("smote", SMOTENC(categorical_features=categorical_feature_indices, random_state=42)),
        ("xgb", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
            tree_method="hist"
        ))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ======================
    # Optuna objective (F1 최적화)
    # ======================
    def objective(trial):
        params = {
            "smote__sampling_strategy": trial.suggest_float("smote__sampling_strategy", 0.1, 1.0),
            "smote__k_neighbors": trial.suggest_int("smote__k_neighbors", 3, 7),
            "xgb__n_estimators": trial.suggest_int("xgb__n_estimators", 200, 800),
            "xgb__learning_rate": trial.suggest_float("xgb__learning_rate", 0.01, 0.3),
            "xgb__max_depth": trial.suggest_int("xgb__max_depth", 3, 8),
            "xgb__min_child_weight": trial.suggest_int("xgb__min_child_weight", 1, 6),
            "xgb__subsample": trial.suggest_float("xgb__subsample", 0.6, 1.0),
            "xgb__colsample_bytree": trial.suggest_float("xgb__colsample_bytree", 0.6, 1.0),
            "xgb__gamma": trial.suggest_float("xgb__gamma", 0.0, 0.5),
            "xgb__reg_lambda": trial.suggest_float("xgb__reg_lambda", 0.0, 5.0),
            "xgb__reg_alpha": trial.suggest_float("xgb__reg_alpha", 0.0, 1.0),
            "xgb__scale_pos_weight": trial.suggest_float("xgb__scale_pos_weight", 1.0, 5.0),
        }
        pipe.set_params(**params)

        # 교차검증 점수 계산
        f1     = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1).mean()
        recall = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="recall", n_jobs=-1).mean()
        acc    = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1).mean()

        trial.set_user_attr("recall", recall)
        trial.set_user_attr("accuracy", acc)

        return f1   # F1을 최적화 기준으로 사용

    # ======================
    # Optuna 실행
    # ======================
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    # ======================
    # Top 10 출력 (F1 기준)
    # ======================
#     trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:10]
#     print("\n===== Top 10 Trials (F1 기준, Recall/Acc 포함) =====")
#     for rank, t in enumerate(trials, 1):
#         print(f"Rank {rank} | F1={t.value:.4f} | Recall={t.user_attrs['recall']:.4f} | "
#               f"Acc={t.user_attrs['accuracy']:.4f} | Params={t.params}")
# ##################################################################################

# ===== Top 10 Trials (F1 기준, Recall/Acc 포함) =====
# Rank 1 | F1=0.9459 | Recall=0.9475 | Acc=0.9952 | Params={'smote__sampling_strategy': 0.16346631910072434, 'smote__k_neighbors': 3, 'xgb__n_estimators': 460, 'xgb__learning_rate': 0.08935840139455938, 'xgb__max_depth': 8, 'xgb__min_child_weight': 6, 'xgb__subsample': 0.9431453956923186, 'xgb__colsample_bytree': 0.6062290898399862, 'xgb__gamma': 0.00815174533087204, 'xgb__reg_lambda': 0.3830679965161744, 'xgb__reg_alpha': 0.08632383505701345, 'xgb__scale_pos_weight': 4.1944855263118965}
# Rank 2 | F1=0.9454 | Recall=0.9503 | Acc=0.9951 | Params={'smote__sampling_strategy': 0.2647853416814887, 'smote__k_neighbors': 4, 'xgb__n_estimators': 628, 'xgb__learning_rate': 0.05086252000356245, 'xgb__max_depth': 8, 'xgb__min_child_weight': 1, 'xgb__subsample': 0.8530094930060577, 'xgb__colsample_bytree': 0.6182800176898097, 'xgb__gamma': 0.1750937208548009, 'xgb__reg_lambda': 2.4327841540573987, 'xgb__reg_alpha': 0.4389356144876564, 'xgb__scale_pos_weight': 3.212537709007618}
# Rank 3 | F1=0.9452 | Recall=0.9479 | Acc=0.9951 | Params={'smote__sampling_strategy': 0.1284943987204762, 'smote__k_neighbors': 4, 'xgb__n_estimators': 331, 'xgb__learning_rate': 0.070314810049945, 'xgb__max_depth': 7, 'xgb__min_child_weight': 6, 'xgb__subsample': 0.8601339796539915, 'xgb__colsample_bytree': 0.6222087978896433, 'xgb__gamma': 0.09925811201549245, 'xgb__reg_lambda': 1.356828731141754, 'xgb__reg_alpha': 0.2255163401252792, 'xgb__scale_pos_weight': 3.843728646907586}
# Rank 4 | F1=0.9452 | Recall=0.9418 | Acc=0.9951 | Params={'smote__sampling_strategy': 0.12035733123536313, 'smote__k_neighbors': 6, 'xgb__n_estimators': 708, 'xgb__learning_rate': 0.100375236661589, 'xgb__max_depth': 8, 'xgb__min_child_weight': 6, 'xgb__subsample': 0.8805086460990815, 'xgb__colsample_bytree': 0.6032108759975742, 'xgb__gamma': 0.19748926599388644, 'xgb__reg_lambda': 0.9833784930264604, 'xgb__reg_alpha': 0.41362069103775717, 'xgb__scale_pos_weight': 3.4318480614917295}
# Rank 5 | F1=0.9449 | Recall=0.9424 | Acc=0.9951 | Params={'smote__sampling_strategy': 0.11957923916825053, 'smote__k_neighbors': 4, 'xgb__n_estimators': 505, 'xgb__learning_rate': 0.09787851387885664, 'xgb__max_depth': 8, 'xgb__min_child_weight': 6, 'xgb__subsample': 0.9018313746199242, 'xgb__colsample_bytree': 0.7032072378265427, 'xgb__gamma': 0.07430844605254956, 'xgb__reg_lambda': 0.4363564683238788, 'xgb__reg_alpha': 0.27793780549896613, 'xgb__scale_pos_weight': 4.157449425098291}
# Rank 6 | F1=0.9447 | Recall=0.9494 | Acc=0.9951 | Params={'smote__sampling_strategy': 0.1519395847700589, 'smote__k_neighbors': 4, 'xgb__n_estimators': 311, 'xgb__learning_rate': 0.08376632770582448, 'xgb__max_depth': 8, 'xgb__min_child_weight': 6, 'xgb__subsample': 0.8521305239708633, 'xgb__colsample_bytree': 0.7105937980465403, 'xgb__gamma': 0.07582422064647112, 'xgb__reg_lambda': 0.4642275611966279, 'xgb__reg_alpha': 0.08323413805750812, 'xgb__scale_pos_weight': 4.132514598002562}
# Rank 7 | F1=0.9446 | Recall=0.9414 | Acc=0.9951 | Params={'smote__sampling_strategy': 0.16302865631007024, 'smote__k_neighbors': 6, 'xgb__n_estimators': 675, 'xgb__learning_rate': 0.11019777407984895, 'xgb__max_depth': 8, 'xgb__min_child_weight': 6, 'xgb__subsample': 0.8751331697953804, 'xgb__colsample_bytree': 0.6141143801148696, 'xgb__gamma': 0.19859275725297967, 'xgb__reg_lambda': 0.7491717064284391, 'xgb__reg_alpha': 0.42506753422232135, 'xgb__scale_pos_weight': 3.2945364391239647}
# Rank 8 | F1=0.9442 | Recall=0.9402 | Acc=0.9951 | Params={'smote__sampling_strategy': 0.1506662415897263, 'smote__k_neighbors': 5, 'xgb__n_estimators': 749, 'xgb__learning_rate': 0.1466044022919137, 'xgb__max_depth': 8, 'xgb__min_child_weight': 5, 'xgb__subsample': 0.8596388352295404, 'xgb__colsample_bytree': 0.6003412408537206, 'xgb__gamma': 0.17738524290110003, 'xgb__reg_lambda': 0.8793400855645954, 'xgb__reg_alpha': 0.24432824229826894, 'xgb__scale_pos_weight': 3.622805987134144}
# Rank 9 | F1=0.9430 | Recall=0.9405 | Acc=0.9949 | Params={'smote__sampling_strategy': 0.10284161210844245, 'smote__k_neighbors': 4, 'xgb__n_estimators': 525, 'xgb__learning_rate': 0.10066625419607746, 'xgb__max_depth': 8, 'xgb__min_child_weight': 6, 'xgb__subsample': 0.895079964474876, 'xgb__colsample_bytree': 0.6370342520385587, 'xgb__gamma': 0.0035021335886248756, 'xgb__reg_lambda': 0.5605493816118909, 'xgb__reg_alpha': 0.3227421954555426, 'xgb__scale_pos_weight': 4.031176311778868}
# Rank 10 | F1=0.9427 | Recall=0.9460 | Acc=0.9949 | Params={'smote__sampling_strategy': 0.15177781077559294, 'smote__k_neighbors': 3, 'xgb__n_estimators': 257, 'xgb__learning_rate': 0.07810490780002509, 'xgb__max_depth': 7, 'xgb__min_child_weight': 1, 'xgb__subsample': 0.9690386452850898, 'xgb__colsample_bytree': 0.7563749096362815, 'xgb__gamma': 0.05766192954395212, 'xgb__reg_lambda': 3.285509530406217, 'xgb__reg_alpha': 0.7203552143000452, 'xgb__scale_pos_weight': 3.1691113697155093}

X_train.columns

drop_cols = ["id","line","name","mold_name","date","time","registration_time",
                 "year_month","hour","datetime","real_time","working"]