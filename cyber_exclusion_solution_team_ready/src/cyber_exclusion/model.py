from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, log_loss
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


def apply_platt(model: LogisticRegression | None, p: np.ndarray) -> np.ndarray:
    if model is None:
        return np.asarray(p, dtype=float)
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


def conformal_sets(p: np.ndarray, qhat: float | None) -> list[str]:
    if qhat is None or not np.isfinite(qhat):
        return ["{KEEP,EXCLUDE}"] * len(p)
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


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Stable percentile ranks in (0, 1]; appropriate when the metric only uses ordering."""
    return pd.Series(np.asarray(values)).rank(method="average", pct=True).to_numpy(dtype=float)


def optimize_rank_blend(y: np.ndarray, p_tab: np.ndarray, p_txt: np.ndarray) -> tuple[float, float, float]:
    """Choose tabular/text rank-blend weights that maximize OOF Average Precision."""
    r_tab = percentile_rank(p_tab)
    r_txt = percentile_rank(p_txt)
    best_w, best_ap = 1.0, -1.0
    for w in np.linspace(0.0, 1.0, 21):
        score = w * r_tab + (1.0 - w) * r_txt
        ap = average_precision_score(y, score)
        if ap > best_ap:
            best_ap, best_w = float(ap), float(w)
    return best_w, 1.0 - best_w, best_ap

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

    def predict_components(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = sanitize_frame(df, self.feature_cols, self.cat_cols)
        p_tab = self.tabular.predict_proba(X)[:, 1]
        text = df.get("text_blob", df["asset"]).fillna("").astype(str)
        p_txt = self.text_model.predict_proba(self.text_vectorizer.transform(text))[:, 1]
        return p_tab, p_txt

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        p_tab, p_txt = self.predict_components(df)
        return self.tabular_weight * p_tab + self.text_weight * p_txt

    def predict_competition_score(self, df: pd.DataFrame) -> np.ndarray:
        # Average Precision only consumes ordering. Percentile-rank each branch so
        # one model's probability scale cannot dominate the ensemble.
        p_tab, p_txt = self.predict_components(df)
        return self.tabular_weight * percentile_rank(p_tab) + self.text_weight * percentile_rank(p_txt)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        raw = self.predict_raw(df)
        p = apply_platt(self.calibrator, raw)
        y_pred = self.predict_competition_score(df)
        result = df[["client_id", "scan_id", "asset"]].copy()
        result["y_pred"] = y_pred
        result["p_exclude"] = p
        result["competition_prediction"] = (p >= self.competition_threshold).astype(int)
        sets = conformal_sets(p, self.conformal_qhat)
        result["conformal_set"] = sets
        decision = np.full(len(df), "REVIEW", dtype=object)
        decision[(p >= self.auto_exclude_threshold) & (np.array(sets) == "{EXCLUDE}")] = "EXCLUDE"
        decision[(p <= self.auto_keep_threshold) & (np.array(sets) == "{KEEP}")] = "KEEP"
        result["decision"] = decision
        result["trust_validated"] = bool(self.metadata.get("trust_validated", True))
        return result

    def save(self, path: str) -> None:
        joblib.dump(self, path)


def train_grouped_ensemble(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[EnsembleBundle, pd.DataFrame, dict[str, Any]]:
    """Train the ensemble.

    When positives span at least two client groups, build leakage-safe grouped OOF
    predictions and use them for AP blend selection/calibration. When they do not,
    still fit a competition ranking model on all available labeled rows, but mark
    every trust/calibration artifact as unvalidated and force production decisions
    to REVIEW. This keeps the competition workflow usable without pretending that
    an honest OOF estimate exists.
    """
    y_s, label_col = _label_from_df(df, cfg["label_candidates"], cfg["positive_values"])
    y = y_s.to_numpy()
    features, cat_cols = select_features(df, label_col)
    X = sanitize_frame(df, features, cat_cols)
    text_values = df.get("text_blob", df["asset"]).fillna("").astype(str).to_numpy()
    groups = df["client_id"].astype(str).to_numpy()

    if len(np.unique(y)) < 2:
        raise ValueError("Training data must contain both excluded=0 and excluded=1 examples.")

    model_cfg = cfg.get("model", {})
    seed = int(cfg.get("random_seed", 42))

    # Always fit a final competition model on all labeled data.
    final_tab = CatBoostClassifier(random_seed=seed, auto_class_weights="Balanced", **model_cfg)
    final_tab.fit(X, y, cat_features=[X.columns.get_loc(c) for c in cat_cols])
    final_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=80000, sublinear_tf=True)
    xt = final_vec.fit_transform(text_values)
    final_lr = LogisticRegression(max_iter=1500, class_weight="balanced", C=2.0)
    final_lr.fit(xt, y)

    positive_groups = np.unique(groups[y == 1])
    negative_groups = np.unique(groups[y == 0])
    grouped_validation_possible = len(positive_groups) >= 2 and len(negative_groups) >= 2

    if not grouped_validation_possible:
        # No honest client-grouped OOF estimate is possible. Use configured blend
        # weights for competition ranking only; do not calibrate or auto-action.
        tw = float(cfg.get("blend", {}).get("tabular_weight", 0.8))
        xw = float(cfg.get("blend", {}).get("text_weight", 0.2))
        z = max(tw + xw, 1e-9)
        tw, xw = tw / z, xw / z

        p_tab = final_tab.predict_proba(X)[:, 1]
        p_txt = final_lr.predict_proba(final_vec.transform(text_values))[:, 1]
        training_rank_score = tw * percentile_rank(p_tab) + xw * percentile_rank(p_txt)
        raw = tw * p_tab + xw * p_txt

        metrics: dict[str, Any] = {
            "evaluation_mode": "in_sample_unvalidated",
            "trust_validated": False,
            "grouped_oof_available": False,
            "average_precision": None,
            "training_in_sample_average_precision": float(average_precision_score(y, training_rank_score)),
            "rank_blend_tabular_weight": float(tw),
            "rank_blend_text_weight": float(xw),
            "positive_rate": float(y.mean()),
            "positive_rows": int(y.sum()),
            "positive_clients": int(len(positive_groups)),
            "negative_clients": int(len(negative_groups)),
            "validation_warning": (
                "No leakage-safe client-grouped OOF Average Precision is available because positives "
                "do not span at least two client_id groups. In-sample AP is diagnostic only."
            ),
        }
        bundle = EnsembleBundle(
            final_tab, final_vec, final_lr, None, features, cat_cols,
            tw, xw, 0.5, 1.0, 0.0, float("nan"), label_col, metrics.copy()
        )
        diag = df[["client_id", "scan_id", "asset"]].copy()
        diag["label"] = y
        diag["y_pred"] = training_rank_score
        diag["p_exclude_uncalibrated"] = raw
        diag["p_tabular"] = p_tab
        diag["p_text"] = p_txt
        diag["is_oof"] = False
        diag["conformal_set"] = "{KEEP,EXCLUDE}"
        return bundle, diag, metrics

    n_splits = min(int(cfg.get("cv_folds", 5)), len(np.unique(groups)), len(positive_groups), len(negative_groups))
    if n_splits < 2:
        raise ValueError("Need at least two feasible grouped folds containing both classes for leakage-safe CV.")

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_tab = np.zeros(len(df))
    oof_txt = np.zeros(len(df))

    for fold, (tr, va) in enumerate(cv.split(X, y, groups), 1):
        if len(np.unique(y[tr])) < 2:
            raise ValueError(f"Fold {fold} training split has only one class; grouped OOF is not feasible.")
        tab = CatBoostClassifier(random_seed=seed + fold, auto_class_weights="Balanced", **model_cfg)
        tab.fit(X.iloc[tr], y[tr], cat_features=[X.columns.get_loc(c) for c in cat_cols])
        oof_tab[va] = tab.predict_proba(X.iloc[va])[:, 1]

        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=80000, sublinear_tf=True)
        xtr = vec.fit_transform(text_values[tr])
        xva = vec.transform(text_values[va])
        lr = LogisticRegression(max_iter=1500, class_weight="balanced", C=2.0)
        lr.fit(xtr, y[tr])
        oof_txt[va] = lr.predict_proba(xva)[:, 1]

    if cfg.get("blend", {}).get("optimize_for_average_precision", True):
        tw, xw, _ = optimize_rank_blend(y, oof_tab, oof_txt)
    else:
        tw = float(cfg.get("blend", {}).get("tabular_weight", 0.8))
        xw = float(cfg.get("blend", {}).get("text_weight", 0.2))
        z = max(tw + xw, 1e-9)
        tw, xw = tw / z, xw / z

    oof_raw = tw * oof_tab + xw * oof_txt
    oof_rank_score = tw * percentile_rank(oof_tab) + xw * percentile_rank(oof_txt)
    calibrator = fit_platt(oof_raw, y)
    oof_p = apply_platt(calibrator, oof_raw)
    comp_t, comp_f1 = optimize_f1_threshold(y, oof_p)
    biz = cfg.get("business", {})
    ex_t = threshold_for_precision(y, oof_p, float(biz.get("auto_exclude_precision_target", 0.98)))
    keep_t = threshold_for_npv(y, oof_p, float(biz.get("auto_keep_npv_target", 0.995)))
    qhat = conformal_qhat(y, oof_p, float(biz.get("conformal_alpha", 0.05)))

    metrics = {
        "evaluation_mode": "client_grouped_oof",
        "trust_validated": True,
        "grouped_oof_available": True,
        "roc_auc": float(roc_auc_score(y, oof_p)) if len(np.unique(y)) > 1 else float("nan"),
        "average_precision": float(average_precision_score(y, oof_rank_score)),
        "calibrated_average_precision": float(average_precision_score(y, oof_p)),
        "rank_blend_tabular_weight": float(tw),
        "rank_blend_text_weight": float(xw),
        "log_loss": float(log_loss(y, np.clip(oof_p, 1e-6, 1-1e-6))),
        "best_f1": float(comp_f1),
        "competition_threshold": float(comp_t),
        "auto_exclude_threshold": float(ex_t),
        "auto_keep_threshold": float(keep_t),
        "conformal_qhat": float(qhat),
        "positive_rate": float(y.mean()),
        "positive_rows": int(y.sum()),
        "positive_clients": int(len(positive_groups)),
    }
    bundle = EnsembleBundle(
        final_tab, final_vec, final_lr, calibrator, features, cat_cols,
        tw, xw, comp_t, ex_t, keep_t, qhat, label_col, metrics.copy()
    )
    oof = df[["client_id", "scan_id", "asset"]].copy()
    oof["label"] = y
    oof["y_pred"] = oof_rank_score
    oof["p_exclude"] = oof_p
    oof["p_tabular"] = oof_tab
    oof["p_text"] = oof_txt
    oof["is_oof"] = True
    oof["conformal_set"] = conformal_sets(oof_p, qhat)
    return bundle, oof, metrics

