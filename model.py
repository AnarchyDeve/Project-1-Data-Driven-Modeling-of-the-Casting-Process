import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. 데이터 불러오기
# -----------------------------
train = pd.read_csv("./data/train.csv")

# 제외할 컬럼
drop_cols = [
    "id","line","name","mold_name",
    "time","date","count","registration_time",
    "tryshot_signal"
]

X = train.drop(columns=drop_cols + ["passorfail"])
y = train["passorfail"]

# -----------------------------
# 2. 수치형 / 범주형 구분
# -----------------------------
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

# -----------------------------
# 3. 전처리 정의
# -----------------------------
num_transformer = SimpleImputer(strategy="mean")
cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ]
)

# -----------------------------
# 4. 모델 정의 (랜덤포레스트)
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=300,    # 트리 개수
    max_depth=None,      # 깊이 제한 없음
    min_samples_split=5, # 과적합 방지
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# -----------------------------
# 5. 파이프라인 결합
# -----------------------------
clf = Pipeline(steps=[("preprocessor", preprocessor),
                     ("model", rf)])

# -----------------------------
# 6. 데이터 분리 & 학습
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf.fit(X_train, y_train)

# -----------------------------
# 7. 평가
# -----------------------------
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"랜덤포레스트 정확도: {acc:.3f}")

# -----------------------------
# 8. 모델 저장
# -----------------------------
joblib.dump(clf, "./models/model.pkl")
print("모델 저장 완료: ./models/model.pkl")















import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# -----------------------------
# 1. 데이터 불러오기
# -----------------------------
train = pd.read_csv("./data/train.csv")

# 제외할 컬럼
drop_cols = [
    "id","line","name","mold_name",
    "time","date","count","registration_time",
    "tryshot_signal"
]

# 독립변수(X), 종속변수(y) 선택
X = train.drop(columns=drop_cols + ["passorfail"])  # 23개 컬럼
y = train["passorfail"]

# 결측치 제거 (간단 처리)
X = X.dropna()
y = y.loc[X.index]

# -----------------------------
# 2. 데이터 분리
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3. 모델 학습
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -----------------------------
# 4. 평가
# -----------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"모델 정확도: {acc:.3f}")

# -----------------------------
# 5. 모델 저장
# -----------------------------
joblib.dump(model, "./models/model.pkl")
print("모델 저장 완료: ./models/model.pkl")

import joblib

# 모델 불러오기
model = joblib.load("./simple_model.pkl")

# 모델 정보 확인
print(model)

print("계수:", model.coef_)
print("절편:", model.intercept_)



import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import joblib
import numpy as np

# 모델 불러오기
model = joblib.load("./simple_model.pkl")

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("불량 예측 대시보드"),
    dcc.Input(id="strength", type="number", placeholder="Physical Strength"),
    dcc.Input(id="biscuit", type="number", placeholder="Biscuit Thickness"),
    html.Button("예측하기", id="predict-button"),
    html.Div(id="output")
])

@app.callback(
    Output("output", "children"),
    [Input("predict-button", "n_clicks")],
    [dash.dependencies.State("strength", "value"),
     dash.dependencies.State("biscuit", "value")]
)
def predict(n_clicks, strength, biscuit):
    if n_clicks is None:
        return ""
    X_new = np.array([[strength, biscuit]])
    pred = model.predict(X_new)[0]
    return "✅ 양품" if pred == 0 else "❌ 불량"

if __name__ == "__main__":
    app.run_server(debug=True)


model = joblib.load("./simple_model.pkl")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib

# -----------------------------
# 1. 데이터 불러오기
# -----------------------------
train = pd.read_csv("./data/train.csv")

# -----------------------------
# 2. Feature / Target 분리
# -----------------------------
drop_cols = ["id", "line", "name", "mold_name", "date", "time", "count", "passorfail"]
X = train.drop(columns=drop_cols)
y = train["passorfail"]

# -----------------------------
# 3. 결측치 처리 + 전처리 정의
# -----------------------------
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

# -----------------------------
# 4. 모델 파이프라인 구성
# -----------------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# -----------------------------
# 5. 데이터 분리 & 학습
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# 6. 평가
# -----------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"모델 정확도: {acc:.3f}")

# -----------------------------
# 7. 모델 저장
# -----------------------------
joblib.dump(model, "./models/full_model.pkl")
print("모델이 full_model.pkl 로 저장되었습니다.")
