from __future__ import annotations
import numpy as np
import pandas as pd

EVIDENCE_FAMILIES = {
    "network": ["asn", "whois_org", "ip_prefix24"],
    "dns": ["cname_reg_domain", "mx_reg_domain", "ns_reg_domain", "spf_present", "dmarc_present"],
    "tls": ["cert_org", "cert_fingerprint", "cert_name_count"],
    "service": ["open_port_count", "protocols", "software_count"],
    "relational": ["scan_consensus_mean", "scan_outlier_score"],
}


def evidence_coverage(df: pd.DataFrame) -> pd.Series:
    scores = []
    for _, row in df.iterrows():
        fam_ok = 0
        for cols in EVIDENCE_FAMILIES.values():
            found = False
            for c in cols:
                if c not in row:
                    continue
                v = row[c]
                if isinstance(v, str) and v.strip(): found = True
                elif isinstance(v, (int, float, np.number)) and float(v) != 0.0: found = True
            fam_ok += int(found)
        scores.append(fam_ok / len(EVIDENCE_FAMILIES))
    return pd.Series(scores, index=df.index, name="evidence_coverage")


def attach_trust_fields(features: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    out["evidence_coverage"] = evidence_coverage(features).values
    out["model_margin"] = np.abs(out["p_exclude"] - 0.5) * 2
    out["trust_score"] = (0.65 * out["model_margin"] + 0.35 * out["evidence_coverage"]).clip(0, 1)
    out["review_reason"] = ""
    out.loc[out["evidence_coverage"] < 0.4, "review_reason"] += "LOW_EVIDENCE;"
    out.loc[out["conformal_set"] == "{KEEP,EXCLUDE}", "review_reason"] += "MODEL_AMBIGUITY;"
    return out
