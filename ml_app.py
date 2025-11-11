import joblib
import pandas as pd
from typing import List, Tuple, Dict, Any
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_PATH = "XGBoost_Optuna.joblib"
CAT_COLS = ["order_protocol", "market_id", "time_category"]  # konsisten seperti training

#load
def load_pipeline(path: str = MODEL_PATH):
    return joblib.load(path)

#schema
def _find_ct(pipe) -> ColumnTransformer:
    for _, step in getattr(pipe, "named_steps", {}).items():
        if isinstance(step, ColumnTransformer):
            return step
    for _, step in getattr(pipe, "steps", []):
        if isinstance(step, ColumnTransformer):
            return step
    raise RuntimeError("ColumnTransformer tidak ditemukan di pipeline.")

def get_expected_columns(pipe) -> List[str]:
    ct = _find_ct(pipe)
    if hasattr(ct, "feature_names_in_"):
        return list(ct.feature_names_in_)
    raise RuntimeError("feature_names_in_ tidak tersedia pada ColumnTransformer.")

def ensure_schema(df: pd.DataFrame, expected_cols: List[str]) -> pd.DataFrame:
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib hilang: {missing}")
    return df[expected_cols]

#full predict
def predict_df(df: pd.DataFrame) -> pd.Series:
    pipe = load_pipeline()
    expected = get_expected_columns(pipe)
    X = ensure_schema(df, expected)
    cat_present = [c for c in CAT_COLS if c in X.columns]
    if cat_present:
        X[cat_present] = X[cat_present].astype("category")
    return pd.Series(pipe.predict(X), index=X.index, name="eta_minutes")

#utilities
def _introspect_cols(ct: ColumnTransformer) -> Tuple[list, list, StandardScaler, OneHotEncoder]:
    num_cols, cat_cols = [], []
    num_scaler = None
    ohe = None
    for _, trans, cols in ct.transformers_:
        if trans is None or cols is None:
            continue
        final = trans.steps[-1][1] if hasattr(trans, "steps") else trans
        if isinstance(final, StandardScaler):
            num_cols = list(cols); num_scaler = final
        elif isinstance(final, OneHotEncoder):
            cat_cols = list(cols); ohe = final
    return num_cols, cat_cols, num_scaler, ohe

def build_defaults_from_model(pipe) -> Tuple[Dict[str, Any], list, list]:
    ct = _find_ct(pipe)
    num_cols, cat_cols, num_scaler, ohe = _introspect_cols(ct)
    defaults: Dict[str, Any] = {}

    if num_scaler is not None and hasattr(num_scaler, "mean_"):
        for col, mu in zip(num_cols, num_scaler.mean_):
            defaults[col] = float(mu)

    if ohe is not None and hasattr(ohe, "categories_"):
        for col, cats in zip(cat_cols, ohe.categories_):
            defaults[col] = cats[0] if len(cats) else None

    return defaults, num_cols, cat_cols

def _derive_features(row: Dict[str, Any]) -> Dict[str, Any]:
    top = max(int(row.get("total_onshift_partners", 0)), 0)
    tbp = max(int(row.get("total_busy_partners", 0)), 0)
    too = max(int(row.get("total_outstanding_orders", 0)), 0)

    available = max(top - tbp, 0)
    row["available_partners"] = available
    row["busy_ratio"] = (tbp / top) if top > 0 else 0.0
    row["order_per_partner"] = (too / available) if available > 0 else 0.0
    row["demand_supply_ratio"] = (too / top) if top > 0 else 0.0

    ti = max(int(row.get("total_items", 0)), 0)
    subtotal = float(row.get("subtotal", 0.0))
    row["avg_price_per_item"] = (subtotal / ti) if ti > 0 else 0.0
    row["is_single_item"] = int(ti == 1)

    dow = int(row.get("created_dayofweek", 0))
    row["is_weekend"] = int(dow in {5, 6})

    # time_category fallback
    if not row.get("time_category"):
        hour = int(row.get("created_hour", 0))
        if   5 <= hour <= 10: row["time_category"] = "morning"
        elif 11 <= hour <= 15: row["time_category"] = "afternoon"
        elif 16 <= hour <= 20: row["time_category"] = "evening"
        else: row["time_category"] = "night"

    return row

def adapt_partial_to_full(partial: Dict[str, Any], pipe) -> pd.DataFrame:
    defaults, _, cat_cols = build_defaults_from_model(pipe)
    row = dict(defaults)
    row.update({k: v for k, v in partial.items() if v is not None})
    row = _derive_features(row)

    ct = _find_ct(pipe)
    expected = list(ct.feature_names_in_) if hasattr(ct, "feature_names_in_") else list(row.keys())
    df = pd.DataFrame([{c: row.get(c, None) for c in expected}])

    cat_present = [c for c in cat_cols if c in df.columns]
    if cat_present:
        df[cat_present] = df[cat_present].astype("category")
    return df[expected]

#predict
def predict_from_partial(partial: Dict[str, Any]) -> float:
    pipe = load_pipeline()
    X = adapt_partial_to_full(partial, pipe)
    y = pipe.predict(X)
    return float(y[0])
