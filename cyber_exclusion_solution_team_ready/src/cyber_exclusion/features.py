from __future__ import annotations
import ipaddress
import math
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any
import numpy as np
import pandas as pd
from .utils import is_ip, registrable_domain, strings_at_keys, flatten_scalars, tokenize_asset, stable_hash

THIRD_PARTY_HINTS = (
    "cloudflare", "akamai", "fastly", "amazon", "aws", "azure", "microsoft", "google", "gcp",
    "salesforce", "sendgrid", "mailgun", "mimecast", "proofpoint", "zendesk", "hubspot", "shopify",
    "github", "atlassian", "okta", "cloudfront", "heroku", "digitalocean", "linode", "oracle"
)

PORT_TERMS = (".port", "port]")


def _extract_services(raw: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    for path in [("services",), ("host", "services")]:
        cur: Any = raw
        ok = True
        for p in path:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                ok = False
                break
        if ok and isinstance(cur, list):
            candidates.extend(x for x in cur if isinstance(x, dict))
    return candidates


def _all_strings(raw: dict[str, Any], max_n: int = 1500) -> list[str]:
    vals = []
    for _, v in flatten_scalars(raw, max_items=5000):
        if isinstance(v, str) and v.strip():
            vals.append(v.strip())
            if len(vals) >= max_n:
                break
    return vals


def extract_asset_features(asset: str, raw: dict[str, Any], vendor: str = "censys") -> dict[str, Any]:
    asset = str(asset).strip().lower().rstrip(".")
    ip_flag = is_ip(asset)
    feat: dict[str, Any] = {
        "asset": asset,
        "asset_type": "ip" if ip_flag else "domain",
        "vendor": vendor or "unknown",
        "asset_len": len(asset),
        "asset_digit_ratio": sum(ch.isdigit() for ch in asset) / max(1, len(asset)),
        "asset_hyphen_count": asset.count("-"),
        "asset_dot_count": asset.count("."),
        "asset_token_count": len(tokenize_asset(asset)),
        "registrable_domain": registrable_domain(asset),
        "subdomain_depth": 0 if ip_flag else max(0, len(asset.split(".")) - len(registrable_domain(asset).split("."))),
        "raw_available": int(bool(raw)),
    }
    if ip_flag:
        ip = ipaddress.ip_address(asset)
        feat.update({
            "ip_version": ip.version,
            "ip_private": int(ip.is_private),
            "ip_global": int(ip.is_global),
            "ip_multicast": int(ip.is_multicast),
            "ip_reserved": int(ip.is_reserved),
            "ip_prefix24": ".".join(asset.split(".")[:3]) if ip.version == 4 else "",
            "ip_prefix16": ".".join(asset.split(".")[:2]) if ip.version == 4 else "",
        })
    else:
        feat.update({"ip_version": 0, "ip_private": 0, "ip_global": 0, "ip_multicast": 0, "ip_reserved": 0,
                     "ip_prefix24": "", "ip_prefix16": ""})

    if not raw:
        feat.update(_blank_raw_features())
        feat["text_blob"] = asset
        return feat

    services = _extract_services(raw)
    ports, protocols, software = [], [], []
    cert_names, cert_orgs, cert_fps = [], [], []
    banners = []
    for s in services:
        p = s.get("port")
        if isinstance(p, (int, float)):
            ports.append(int(p))
        proto = s.get("protocol") or s.get("service_name") or s.get("extended_service_name")
        if proto:
            protocols.append(str(proto).lower())
        if s.get("banner"):
            banners.append(str(s.get("banner"))[:500])
        for sw in s.get("software", []) if isinstance(s.get("software"), list) else []:
            if isinstance(sw, dict):
                software.append(" ".join(str(sw.get(k, "")) for k in ["vendor", "product", "version"]).strip())

    dns_names = strings_at_keys(raw, ["dns.names", "reverse_dns", "forward_dns", "hostname", "names"])
    whois_orgs = strings_at_keys(raw, ["whois.organization", "whois.network.name", "autonomous_system.name", "organization"])
    asns = strings_at_keys(raw, ["autonomous_system.asn", ".asn"])
    cert_names = strings_at_keys(raw, ["certificate.names", "cert.names", "subject_alt_name.dns_names", "leaf_data.names"])
    cert_orgs = strings_at_keys(raw, ["certificate.parsed.subject.organization", "leaf_data.subject.organization", "subject.organization"])
    cert_fps = strings_at_keys(raw, ["fingerprint_sha256", "leaf_data.fingerprint", "certificate.fingerprint"])
    cname_vals = strings_at_keys(raw, ["cname"])
    mx_vals = strings_at_keys(raw, ["mx", "mail_exchanger"])
    ns_vals = strings_at_keys(raw, ["nameserver", "name_server", " ns", ".ns"])
    txt_vals = strings_at_keys(raw, ["txt", "spf", "dmarc", "dkim"])
    vulns = strings_at_keys(raw, ["cve", "vulnerab"])

    all_strings = _all_strings(raw)
    text = " ".join(all_strings).lower()
    provider_hits = sorted({h for h in THIRD_PARTY_HINTS if h in text})
    domain_names = [x.lower().rstrip(".") for x in dns_names + cert_names + cname_vals + mx_vals + ns_vals if "." in x]
    reg_domains = [registrable_domain(x) for x in domain_names if registrable_domain(x)]
    unrelated_reg = len({d for d in reg_domains if d and d != registrable_domain(asset)})

    feat.update({
        "service_count": len(services),
        "open_port_count": len(set(ports)),
        "open_ports": ",".join(map(str, sorted(set(ports))[:40])),
        "protocol_count": len(set(protocols)),
        "protocols": ",".join(sorted(set(protocols))[:30]),
        "software_count": len(set(x for x in software if x)),
        "dns_name_count": len(set(dns_names)),
        "cert_name_count": len(set(cert_names)),
        "cert_org_count": len(set(cert_orgs)),
        "cert_fp_count": len(set(cert_fps)),
        "whois_org_count": len(set(whois_orgs)),
        "asn_count": len(set(asns)),
        "cname_count": len(set(cname_vals)),
        "mx_count": len(set(mx_vals)),
        "ns_count": len(set(ns_vals)),
        "txt_auth_count": len(set(txt_vals)),
        "spf_present": int("v=spf1" in text),
        "dmarc_present": int("v=dmarc1" in text),
        "dkim_hint": int("dkim" in text or "._domainkey" in text),
        "vulnerability_signal_count": len(set(vulns)),
        "provider_hint_count": len(provider_hits),
        "provider_hints": ",".join(provider_hits),
        "shared_name_pressure": len(set(reg_domains)),
        "unrelated_reg_domain_count": unrelated_reg,
        "whois_org": "|".join(sorted(set(whois_orgs))[:5]),
        "asn": "|".join(sorted(set(asns))[:5]),
        "cert_org": "|".join(sorted(set(cert_orgs))[:5]),
        "cert_fingerprint": "|".join(sorted(set(cert_fps))[:5]),
        "cname_reg_domain": "|".join(sorted({registrable_domain(x) for x in cname_vals if x})[:5]),
        "mx_reg_domain": "|".join(sorted({registrable_domain(x) for x in mx_vals if x})[:5]),
        "ns_reg_domain": "|".join(sorted({registrable_domain(x) for x in ns_vals if x})[:5]),
        "banner_hash": stable_hash("|".join(banners[:5])) if banners else "",
    })
    feat["text_blob"] = " ".join([
        asset, feat["registrable_domain"], feat["whois_org"], feat["asn"], feat["cert_org"],
        feat["provider_hints"], feat["cname_reg_domain"], feat["mx_reg_domain"], feat["ns_reg_domain"],
        " ".join(protocols[:20]), " ".join(software[:20]), " ".join(domain_names[:60]), " ".join(banners[:8])
    ])[:20000]
    return feat


def _blank_raw_features() -> dict[str, Any]:
    return {
        "service_count": 0, "open_port_count": 0, "open_ports": "", "protocol_count": 0, "protocols": "",
        "software_count": 0, "dns_name_count": 0, "cert_name_count": 0, "cert_org_count": 0,
        "cert_fp_count": 0, "whois_org_count": 0, "asn_count": 0, "cname_count": 0, "mx_count": 0,
        "ns_count": 0, "txt_auth_count": 0, "spf_present": 0, "dmarc_present": 0, "dkim_hint": 0,
        "vulnerability_signal_count": 0, "provider_hint_count": 0, "provider_hints": "", "shared_name_pressure": 0,
        "unrelated_reg_domain_count": 0, "whois_org": "", "asn": "", "cert_org": "", "cert_fingerprint": "",
        "cname_reg_domain": "", "mx_reg_domain": "", "ns_reg_domain": "", "banner_hash": ""
    }


def add_scan_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    group_cols = ["client_id", "scan_id"]
    signatures = ["registrable_domain", "asn", "whois_org", "cert_org", "cert_fingerprint", "cname_reg_domain", "mx_reg_domain", "ns_reg_domain", "ip_prefix24"]
    for col in signatures:
        if col not in out:
            continue
        counts = out.groupby(group_cols + [col], dropna=False)[col].transform("size")
        gsize = out.groupby(group_cols)[col].transform("size")
        out[f"scan_{col}_count"] = counts.astype(float)
        out[f"scan_{col}_share"] = (counts / gsize.clip(lower=1)).astype(float)
        mode_map = out.groupby(group_cols)[col].agg(lambda s: s.astype(str).value_counts().index[0] if len(s) else "")
        keys = list(zip(out["client_id"], out["scan_id"]))
        modes = [mode_map.loc[k] for k in keys]
        out[f"scan_{col}_is_mode"] = (out[col].astype(str).values == np.array(modes, dtype=object)).astype(int)
    out["scan_asset_count"] = out.groupby(group_cols)["asset"].transform("size")
    out["scan_ip_fraction"] = out.groupby(group_cols)["asset_type"].transform(lambda s: (s == "ip").mean())

    # Consensus / outlier score: low support across multiple independent signals is suspicious.
    share_cols = [c for c in out.columns if c.startswith("scan_") and c.endswith("_share")]
    if share_cols:
        out["scan_consensus_mean"] = out[share_cols].mean(axis=1)
        out["scan_consensus_min"] = out[share_cols].min(axis=1)
        out["scan_outlier_score"] = 1.0 - out["scan_consensus_mean"]
    else:
        out["scan_consensus_mean"] = 0.0
        out["scan_consensus_min"] = 0.0
        out["scan_outlier_score"] = 1.0
    return out


def add_seed_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "seed_domain" not in out.columns:
        out["seed_domain"] = ""
    seed_reg = out["seed_domain"].fillna("").astype(str).map(registrable_domain)
    asset_reg = out["registrable_domain"].fillna("").astype(str)
    out["same_reg_domain_as_seed"] = (seed_reg != "") & (seed_reg == asset_reg)
    out["same_reg_domain_as_seed"] = out["same_reg_domain_as_seed"].astype(int)
    out["asset_seed_similarity"] = [SequenceMatcher(None, a, b).ratio() if b else 0.0 for a, b in zip(out["asset"].astype(str), out["seed_domain"].astype(str))]
    return out
