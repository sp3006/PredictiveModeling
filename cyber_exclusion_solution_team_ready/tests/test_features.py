from cyber_exclusion.features import extract_asset_features, add_scan_relative_features
import pandas as pd

def test_ip_features():
    raw={"host":{"services":[{"port":443,"protocol":"HTTPS","cert":{"names":["example.com"]}}],"autonomous_system":{"name":"Example Cloud","asn":64500}},"dns":{"names":["a.example.com"]}}
    f=extract_asset_features("3.220.243.127",raw)
    assert f["asset_type"]=="ip"
    assert f["open_port_count"]==1

def test_scan_relative():
    d=pd.DataFrame([
      {"client_id":"c","scan_id":"s","asset":"a.com","asset_type":"domain","registrable_domain":"a.com","asn":"1","whois_org":"x","cert_org":"x","cert_fingerprint":"f","cname_reg_domain":"","mx_reg_domain":"","ns_reg_domain":"x","ip_prefix24":""},
      {"client_id":"c","scan_id":"s","asset":"b.a.com","asset_type":"domain","registrable_domain":"a.com","asn":"1","whois_org":"x","cert_org":"x","cert_fingerprint":"f","cname_reg_domain":"","mx_reg_domain":"","ns_reg_domain":"x","ip_prefix24":""},
    ])
    out=add_scan_relative_features(d)
    assert out["scan_registrable_domain_share"].min()==1.0
