import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


def preprocess(x, y=None, fit_state=None,
               num_imputation_strategy="median",
               cat_encoding_method="woe",
               nan_threshold_high=0.6,
               nan_threshold_low=0.17,
               iv_threshold=0.02,
               corr_threshold=0.6,
               n_shap_features_to_drop=0,
               shap_importance_df=None):

    is_fit = fit_state is None
    if is_fit:
        fit_state = {}

    x = x.copy()

    # ── Time features ────────────────────────────────────────────────────────
    x["DT_day"]        = (x["TransactionDT"] // (3600 * 24)) % 7
    x["DT_hour"]       = (x["TransactionDT"] // 3600) % 24
    x["DT_dayofmonth"] = (x["TransactionDT"] // (3600 * 24)) % 30
    x["DT_month"]      = (x["TransactionDT"] // (3600 * 24)) // 30

    if "TransactionAmt" in x.columns:
        x["TransactionAmt"] = np.log1p(x["TransactionAmt"])

    # ── Drop high-NA columns ─────────────────────────────────────────────────
    if is_fit:
        na_percent = x.isna().mean()
        fit_state["cols_to_drop_high"] = na_percent[na_percent > nan_threshold_high].index.tolist()
    x = x.drop(columns=fit_state["cols_to_drop_high"], errors="ignore")

    # ── IV filter ────────────────────────────────────────────────────────────
    if is_fit:
        iv_values = _calculate_iv(x, y)
        fit_state["cols_low_iv"] = iv_values[iv_values < iv_threshold].index.tolist()
    x = x.drop(columns=fit_state["cols_low_iv"], errors="ignore")

    # ── NaN flags ────────────────────────────────────────────────────────────
    if is_fit:
        na_percent = x.isna().mean()
        fit_state["cols_to_flag"] = na_percent[na_percent > nan_threshold_low].index.tolist()
    nan_flags = pd.concat(
        [x[col].isnull().astype(int).rename(col + "_isNaN")
         for col in fit_state["cols_to_flag"] if col in x.columns],
        axis=1
    )
    x = pd.concat([x, nan_flags], axis=1)

    # ── Numerical imputation ─────────────────────────────────────────────────
    num_cols = x.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        if is_fit:
            if num_imputation_strategy == "mean":
                num_imputer = SimpleImputer(strategy="mean")
            else:
                num_imputer = SimpleImputer(strategy="median")
            num_imputer.fit(x[num_cols])
            fit_state["num_imputer"] = num_imputer
            fit_state["num_cols"] = num_cols
        x[fit_state["num_cols"]] = fit_state["num_imputer"].transform(x[fit_state["num_cols"]])

    x = x.fillna("missing")

    # ── OHE low cardinality ──────────────────────────────────────────────────
    if is_fit:
        nunique = x.select_dtypes(include="object").nunique()
        fit_state["low_card_cols"] = nunique[nunique < 5].index.tolist()
        fit_state["ohe_columns"] = pd.get_dummies(
            x, columns=fit_state["low_card_cols"], drop_first=True, dtype=int
        ).columns.tolist()

    x = pd.get_dummies(x, columns=fit_state["low_card_cols"], drop_first=True, dtype=int)
    x = x.reindex(columns=fit_state["ohe_columns"], fill_value=0)

    # ── Encode high cardinality ──────────────────────────────────────────────
    high_card_cols = x.select_dtypes(include="object").columns.tolist()

    if is_fit:
        fit_state["encoding_maps"] = {}
        for col in high_card_cols:
            if cat_encoding_method == "woe":
                fit_state["encoding_maps"][col] = _calculate_woe(x[col], y)
            elif cat_encoding_method == "mean":
                fit_state["encoding_maps"][col] = x.groupby(col)[y].mean().to_dict()
            elif cat_encoding_method == "median":
                fit_state["encoding_maps"][col] = x.groupby(col)[y].median().to_dict()

    for col in high_card_cols:
        if col in fit_state["encoding_maps"]:
            enc_map = fit_state["encoding_maps"][col]
            x[col] = x[col].map(enc_map).fillna(enc_map.get("missing", 0))

    # ── Correlation filter ───────────────────────────────────────────────────
    if is_fit:
        corr_matrix = np.abs(x.corr())
        cols = corr_matrix.columns.tolist()
        to_drop = set()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if cols[i] in to_drop or cols[j] in to_drop:
                    continue
                if corr_matrix.loc[cols[i], cols[j]] > corr_threshold:
                    corr_i = abs(x[cols[i]].corr(y))
                    corr_j = abs(x[cols[j]].corr(y))
                    if corr_i >= corr_j:
                        to_drop.add(cols[j])
                    else:
                        to_drop.add(cols[i])
        fit_state["cols_to_drop_corr"] = list(to_drop)
    x = x.drop(columns=fit_state["cols_to_drop_corr"], errors="ignore")

    # ── SHAP-based feature dropping ──────────────────────────────────────────
    if is_fit and shap_importance_df is not None:
        fit_state["shap_features_to_drop"] = (
            shap_importance_df.sort_values("importance", ascending=True)["feature"]
            .head(n_shap_features_to_drop)
            .tolist()
        )
    if "shap_features_to_drop" in fit_state:
        x = x.drop(columns=fit_state["shap_features_to_drop"], errors="ignore")

    # ── Drop ID columns ──────────────────────────────────────────────────────
    x = x.drop(columns=[c for c in ["TransactionID", "isFraud"] if c in x.columns],
                errors="ignore")

    # ── Final column alignment ───────────────────────────────────────────────
    if is_fit:
        fit_state["final_columns"] = x.columns.tolist()
    else:
        x = x.reindex(columns=fit_state["final_columns"], fill_value=0)

    return x, y, fit_state


def _calculate_iv(df, y):
    iv_values = {}

    for col in df.columns:
        if col in ["TransactionDT", "TransactionID"]:
            iv_values[col] = 0
            continue

        if df[col].dtype == "object" or df[col].nunique() < 20:
            grouped = df.groupby(col)[y].agg(["sum", "count"])
            grouped.columns = ["events", "total"]
            grouped["non_events"] = grouped["total"] - grouped["events"]

            total_events = grouped["events"].sum()
            total_non_events = grouped["non_events"].sum()

            if total_events == 0 or total_non_events == 0:
                iv_values[col] = 0
                continue

            grouped["pct_events"]     = grouped["events"] / total_events
            grouped["pct_non_events"] = grouped["non_events"] / total_non_events
            grouped["pct_events"]     = grouped["pct_events"].replace(0, 0.0001)
            grouped["pct_non_events"] = grouped["pct_non_events"].replace(0, 0.0001)
            grouped["woe"] = np.log(grouped["pct_events"] / grouped["pct_non_events"])
            grouped["iv"]  = (grouped["pct_events"] - grouped["pct_non_events"]) * grouped["woe"]

            iv_values[col] = grouped["iv"].sum()
        else:
            iv_values[col] = 0

    return pd.Series(iv_values)


def _calculate_woe(feature, target):
    df_temp = pd.DataFrame({"feature": feature, "target": target})
    grouped = df_temp.groupby("feature")["target"].agg(["sum", "count"])
    grouped.columns = ["events", "total"]
    grouped["non_events"] = grouped["total"] - grouped["events"]

    total_events     = grouped["events"].sum()
    total_non_events = grouped["non_events"].sum()

    if total_events == 0 or total_non_events == 0:
        return {cat: 0 for cat in grouped.index}

    grouped["pct_events"]     = grouped["events"] / total_events
    grouped["pct_non_events"] = grouped["non_events"] / total_non_events
    grouped["pct_events"]     = grouped["pct_events"].replace(0, 0.0001)
    grouped["pct_non_events"] = grouped["pct_non_events"].replace(0, 0.0001)
    grouped["woe"] = np.log(grouped["pct_events"] / grouped["pct_non_events"])

    return grouped["woe"].to_dict()