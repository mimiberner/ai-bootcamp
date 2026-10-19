"""Train and compare house-price models on the California Housing dataset.

Outputs results.json, which build_report.py turns into the HTML report.
"""
import json
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

SEED = 42
TARGET = "median_house_value"

df = pd.read_csv("data/housing.csv")

# Engineered ratios: raw totals scale with district size, ratios describe the homes.
df["rooms_per_household"] = df.total_rooms / df.households
df["bedrooms_per_room"] = df.total_bedrooms / df.total_rooms
df["people_per_household"] = df.population / df.households

X = df.drop(columns=TARGET)
y = df[TARGET]
num_cols = X.select_dtypes("number").columns.tolist()
cat_cols = ["ocean_proximity"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)


def signed_log(a):
    return np.sign(a) * np.log1p(np.abs(a))


def preprocessor(scale):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        # Linear models and neural nets are thrown off by long tails
        # (e.g. one district with 1,243 people per household), so compress them.
        num_steps.append(("log", FunctionTransformer(signed_log, feature_names_out="one-to-one")))
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])


Y_MIN, Y_MAX = y.min(), y.max()


def to_price(log_y):
    # Prices in the data are capped at $500,001, so keep predictions within that range.
    return np.clip(np.expm1(log_y), Y_MIN, Y_MAX)


def make(estimator, scale=False):
    # Every model predicts log(price): errors become relative, which suits prices.
    return TransformedTargetRegressor(
        regressor=Pipeline([("prep", preprocessor(scale)), ("model", estimator)]),
        func=np.log1p, inverse_func=to_price, check_inverse=False,
    )


xgb = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
                   colsample_bytree=0.8, n_jobs=-1, random_state=SEED)
lgbm = LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8,
                     subsample_freq=1, colsample_bytree=0.8, n_jobs=-1, random_state=SEED,
                     importance_type="gain", verbose=-1)
rf = RandomForestRegressor(n_estimators=150, min_samples_leaf=2, max_features=0.5,
                           n_jobs=-1, random_state=SEED)

models = {
    "Linear Regression": (make(LinearRegression(), scale=True),
                          "Ordinary least squares on scaled features"),
    "Decision Tree": (make(DecisionTreeRegressor(max_depth=10, min_samples_leaf=5,
                                                 random_state=SEED)),
                      "Single tree, max depth 10"),
    "Random Forest": (make(rf), "150 trees, min 2 samples per leaf"),
    "XGBoost": (make(xgb), "400 boosted trees, depth 6, learning rate 0.05"),
    "Neural Network": (make(MLPRegressor(hidden_layer_sizes=(128, 64), early_stopping=True,
                                         max_iter=300, learning_rate_init=0.002,
                                         alpha=1e-4, random_state=SEED), scale=True),
                       "MLP with 128 → 64 hidden units, early stopping"),
    "LightGBM": (make(lgbm), "600 boosted trees, 63 leaves, learning rate 0.05"),
    "Stacked Ensemble": (make(StackingRegressor(
        estimators=[("rf", rf), ("xgb", xgb), ("lgbm", lgbm)],
        final_estimator=RidgeCV(), cv=5, n_jobs=1)),
        "Random Forest + XGBoost + LightGBM, blended by a ridge regression"),
}

cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
results = []
for name, (model, desc) in models.items():
    print(f"Training {name}...", flush=True)
    cv_rmse = -cross_val_score(model, X_train, y_train, cv=cv,
                               scoring="neg_root_mean_squared_error", n_jobs=1)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = model.predict(X_test)
    pred_s = time.perf_counter() - t0
    train_pred = model.predict(X_train)

    ape = np.abs(pred - y_test) / y_test
    res = {
        "name": name,
        "description": desc,
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "mae": float(mean_absolute_error(y_test, pred)),
        "medae": float(np.median(np.abs(pred - y_test))),
        "mape": float(ape.mean() * 100),
        "within10": float((ape <= 0.10).mean() * 100),
        "r2": float(r2_score(y_test, pred)),
        "r2_train": float(r2_score(y_train, train_pred)),
        "cv_rmse_mean": float(cv_rmse.mean()),
        "cv_rmse_std": float(cv_rmse.std()),
        "fit_seconds": fit_s,
        "predict_ms_per_1k": pred_s / len(X_test) * 1000 * 1000,
    }
    # Fixed sample of test points for the predicted-vs-actual chart.
    idx = np.random.default_rng(SEED).choice(len(y_test), 600, replace=False)
    res["sample"] = [[float(y_test.iloc[i]), float(pred[i])] for i in idx]

    inner = model.regressor_.named_steps["model"]
    feat_names = model.regressor_.named_steps["prep"].get_feature_names_out()
    feat_names = [f.split("__", 1)[1] for f in feat_names]
    if hasattr(inner, "feature_importances_"):
        imp = inner.feature_importances_.astype(float)
        imp = imp / imp.sum()
        res["importance"] = sorted(zip(feat_names, imp.tolist()), key=lambda t: -t[1])
    results.append(res)
    print(f"  RMSE {res['rmse']:,.0f}  R2 {res['r2']:.3f}  fit {fit_s:.1f}s", flush=True)

summary = {
    "dataset": {
        "name": "California Housing (1990 US Census)",
        "rows": len(df), "train_rows": len(X_train), "test_rows": len(X_test),
        "features": num_cols + cat_cols,
        "target_median": float(y.median()),
        "target_mean": float(y.mean()),
        "capped_share": float((y >= 500000).mean() * 100),
    },
    "results": results,
}
with open("results.json", "w") as f:
    json.dump(summary, f)
print("Saved results.json")
