from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, precision_recall_curve, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.isotonic import IsotonicRegression
import joblib

ID_COLS = {"client_id", "scan_id", "asset", "raw_path", "text_blob"}


def _label_from_df(df: pd.DataFrame, label_candidates: list[str], positive_values: list[Any]) -> tuple[pd.Series, str]:
    lower = {str(v).lower() for v in positive_values}
    for c in label_candidates:
        if c in df.columns:
            s = df[c]
            if pd.api.types.is_numeric_dtype(s):
                y = (pd.to_numeric(s, errors="coerce").fillna(0) > 0).astype(int)
            else:
                y = s.astype(str).str.lower().isin(lower).astype(int)
            return y, c
    raise ValueError(f"No label column found. Tried {label_candidates}")


def select_features(df: pd.DataFrame, label_col: str | None = None) -> tuple[list[str], list[str]]:
    excluded = set(ID_COLS)
    if label_col:
        excluded.add(label_col)
    features = [c for c in df.columns if c not in excluded and not c.startswith("prediction_")]
    cat_cols = [c for c in features if df[c].dtype == "object" or str(df[c].dtype).startswith("bool")]
    return features, cat_cols


def sanitize_frame(df: pd.DataFrame, feature_cols: list[str], cat_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols].copy()
    for c in feature_cols:
        if c in cat_cols:
            X[c] = X[c].fillna("").astype(str)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X


def fit_platt(oof_p: np.ndarray, y: np.ndarray) -> LogisticRegression:
    eps = 1e-6
    logits = np.log(np.clip(oof_p, eps, 1-eps) / np.clip(1-oof_p, eps, 1-eps)).reshape(-1, 1)
    lr = LogisticRegression(C=1.0, max_iter=2000)
    lr.fit(logits, y)
    return lr


def apply_platt(model: LogisticRegression, p: np.ndarray) -> np.ndarray:
    eps = 1e-6
    logits = np.log(np.clip(p, eps, 1-eps) / np.clip(1-p, eps, 1-eps)).reshape(-1, 1)
    return model.predict_proba(logits)[:, 1]


def optimize_f1_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    best_t, best = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        score = f1_score(y, p >= t)
        if score > best:
            best, best_t = float(score), float(t)
    return best_t, best


def threshold_for_precision(y: np.ndarray, p: np.ndarray, target_precision: float) -> float:
    order = np.argsort(-p)
    ys = y[order]
    ps = p[order]
    tp = np.cumsum(ys)
    pred = np.arange(1, len(ys)+1)
    precision = tp / pred
    valid = np.where(precision >= target_precision)[0]
    if not len(valid):
        return 1.0
    i = valid[-1]
    return float(ps[i])


def threshold_for_npv(y: np.ndarray, p: np.ndarray, target_npv: float) -> float:
    order = np.argsort(p)
    ys = y[order]
    ps = p[order]
    tn = np.cumsum(1 - ys)
    pred_neg = np.arange(1, len(ys)+1)
    npv = tn / pred_neg
    valid = np.where(npv >= target_npv)[0]
    if not len(valid):
        return 0.0
    i = valid[-1]
    return float(ps[i])


def conformal_qhat(y: np.ndarray, p: np.ndarray, alpha: float = 0.05) -> float:
    p_true = np.where(y == 1, p, 1-p)
    scores = 1 - p_true
    n = len(scores)
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, q_level, method="higher"))


def conformal_sets(p: np.ndarray, qhat: float) -> list[str]:
    out = []
    for pi in p:
        include0 = (1 - (1-pi)) <= qhat  # nonconformity for class 0 = pi
        include1 = (1 - pi) <= qhat
        if include0 and include1:
            out.append("{KEEP,EXCLUDE}")
        elif include1:
            out.append("{EXCLUDE}")
        elif include0:
            out.append("{KEEP}")
        else:
            out.append("{}")
    return out

@dataclass
class EnsembleBundle:
    tabular: Any
    text_vectorizer: Any
    text_model: Any
    calibrator: Any
    feature_cols: list[str]
    cat_cols: list[str]
    tabular_weight: float
    text_weight: float
    competition_threshold: float
    auto_exclude_threshold: float
    auto_keep_threshold: float
    conformal_qhat: float
    label_col: str
    metadata: dict[str, Any]

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        X = sanitize_frame(df, self.feature_cols, self.cat_cols)
        p_tab = self.tabular.predict_proba(X)[:, 1]
        text = df.get("text_blob", df["asset"]).fillna("").astype(str)
        p_txt = self.text_model.predict_proba(self.text_vectorizer.transform(text))[:, 1]
        return self.tabular_weight * p_tab + self.text_weight * p_txt

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        raw = self.predict_raw(df)
        p = apply_platt(self.calibrator, raw)
        result = df[["client_id", "scan_id", "asset"]].copy()
        result["p_exclude"] = p
        result["competition_prediction"] = (p >= self.competition_threshold).astype(int)
        sets = conformal_sets(p, self.conformal_qhat)
        result["conformal_set"] = sets
        decision = np.full(len(df), "REVIEW", dtype=object)
        decision[(p >= self.auto_exclude_threshold) & (np.array(sets) == "{EXCLUDE}")] = "EXCLUDE"
        decision[(p <= self.auto_keep_threshold) & (np.array(sets) == "{KEEP}")] = "KEEP"
        result["decision"] = decision
        return result

    def save(self, path: str) -> None:
        joblib.dump(self, path)


def train_grouped_ensemble(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[EnsembleBundle, pd.DataFrame, dict[str, float]]:
    y_s, label_col = _label_from_df(df, cfg["label_candidates"], cfg["positive_values"])
    y = y_s.to_numpy()
    features, cat_cols = select_features(df, label_col)
    X = sanitize_frame(df, features, cat_cols)
    text = df.get("text_blob", df["asset"]).fillna("").astype(str).to_numpy()
    groups = df["client_id"].astype(str).to_numpy()
    n_splits = min(int(cfg.get("cv_folds", 5)), len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("Need at least two distinct client_id groups for leakage-safe CV.")
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(cfg.get("random_seed", 42)))
    oof_tab = np.zeros(len(df))
    oof_txt = np.zeros(len(df))
    model_cfg = cfg.get("model", {})

    for fold, (tr, va) in enumerate(cv.split(X, y, groups), 1):
        tab = CatBoostClassifier(random_seed=int(cfg.get("random_seed", 42)) + fold, auto_class_weights="Balanced", **model_cfg)
        tab.fit(X.iloc[tr], y[tr], cat_features=[X.columns.get_loc(c) for c in cat_cols])
        oof_tab[va] = tab.predict_proba(X.iloc[va])[:, 1]

        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=80000, sublinear_tf=True)
        xtr = vec.fit_transform(text[tr])
        xva = vec.transform(text[va])
        lr = LogisticRegression(max_iter=1500, class_weight="balanced", C=2.0)
        lr.fit(xtr, y[tr])
        oof_txt[va] = lr.predict_proba(xva)[:, 1]

    tw = float(cfg.get("blend", {}).get("tabular_weight", 0.8))
    xw = float(cfg.get("blend", {}).get("text_weight", 0.2))
    s = max(tw + xw, 1e-9); tw, xw = tw/s, xw/s
    oof_raw = tw * oof_tab + xw * oof_txt
    calibrator = fit_platt(oof_raw, y)
    oof_p = apply_platt(calibrator, oof_raw)
    comp_t, comp_f1 = optimize_f1_threshold(y, oof_p)
    biz = cfg.get("business", {})
    ex_t = threshold_for_precision(y, oof_p, float(biz.get("auto_exclude_precision_target", 0.98)))
    keep_t = threshold_for_npv(y, oof_p, float(biz.get("auto_keep_npv_target", 0.995)))
    qhat = conformal_qhat(y, oof_p, float(biz.get("conformal_alpha", 0.05)))

    final_tab = CatBoostClassifier(random_seed=int(cfg.get("random_seed", 42)), auto_class_weights="Balanced", **model_cfg)
    final_tab.fit(X, y, cat_features=[X.columns.get_loc(c) for c in cat_cols])
    final_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=80000, sublinear_tf=True)
    xt = final_vec.fit_transform(text)
    final_lr = LogisticRegression(max_iter=1500, class_weight="balanced", C=2.0)
    final_lr.fit(xt, y)

    metrics = {
        "roc_auc": float(roc_auc_score(y, oof_p)) if len(np.unique(y)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y, oof_p)),
        "log_loss": float(log_loss(y, np.clip(oof_p, 1e-6, 1-1e-6))),
        "best_f1": float(comp_f1),
        "competition_threshold": float(comp_t),
        "auto_exclude_threshold": float(ex_t),
        "auto_keep_threshold": float(keep_t),
        "conformal_qhat": float(qhat),
        "positive_rate": float(y.mean()),
    }
    bundle = EnsembleBundle(final_tab, final_vec, final_lr, calibrator, features, cat_cols, tw, xw, comp_t, ex_t, keep_t, qhat, label_col, metrics.copy())
    oof = df[["client_id", "scan_id", "asset"]].copy()
    oof["label"] = y
    oof["p_exclude"] = oof_p
    oof["p_tabular"] = oof_tab
    oof["p_text"] = oof_txt
    oof["conformal_set"] = conformal_sets(oof_p, qhat)
    return bundle, oof, metrics
