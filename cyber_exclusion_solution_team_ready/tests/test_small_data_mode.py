import pandas as pd
from cyber_exclusion.features import add_scan_relative_features, add_seed_features, extract_asset_features
from cyber_exclusion.model import train_grouped_ensemble


def test_small_data_mode_trains_but_disables_trust():
    rows=[]
    for client, scan, assets in [
        ('c1','s1',[('1.1.1.1',0),('2.2.2.2',0)]),
        ('c2','s2',[('3.3.3.3',0),('4.4.4.4',0)]),
        ('c3','s3',[('9.9.9.9',1),('5.5.5.5',0)]),
        ('c3','s4',[('9.9.9.9',1),('6.6.6.6',0)]),
    ]:
        for asset,y in assets:
            rec={'client_id':client,'scan_id':scan,'asset':asset,'excluded':y}
            rec.update(extract_asset_features(asset, {}))
            rows.append(rec)
    df=add_seed_features(add_scan_relative_features(pd.DataFrame(rows)))
    cfg={
        'label_candidates':['excluded'], 'positive_values':[1], 'cv_folds':3, 'random_seed':1,
        'model':{'iterations':10,'depth':3,'learning_rate':0.1,'verbose':False},
        'blend':{'tabular_weight':0.8,'text_weight':0.2,'optimize_for_average_precision':True},
        'business':{'auto_exclude_precision_target':0.98,'auto_keep_npv_target':0.995,'conformal_alpha':0.05},
    }
    bundle, diag, metrics=train_grouped_ensemble(df,cfg)
    assert metrics['grouped_oof_available'] is False
    assert metrics['average_precision'] is None
    assert diag['is_oof'].eq(False).all()
    pred=bundle.predict(df)
    assert pred['decision'].eq('REVIEW').all()
    assert pred['trust_validated'].eq(False).all()
