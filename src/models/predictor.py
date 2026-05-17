# predictor.py — carga el pkl, genera señal + probabilidades + SHAP por fila
#
# SHAP: se usa la API nativa de XGBoost (pred_contribs=True) para evitar
# el bug de incompatibilidad entre shap 0.49 y xgboost 3.x.
# Ver notebooks/_iteraciones/07_post_iter1.ipynb → compute_shap_native()

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings('ignore')

SIGNAL_MAP   = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
POSITION_MAP = {0: -1.0,   1:  0.0,   2:  1.0}


def load_model(path):
    path = Path(path)
    assert path.exists(), f'Modelo no encontrado: {path}'
    cache = pickle.load(open(path, 'rb'))
    assert 'model' in cache and 'feature_cols' in cache, 'pkl inválido: faltan keys'
    print(f'Modelo cargado | tipo={cache["model_type"]} | target={cache["target_col"]} | features={len(cache["feature_cols"])}')
    return cache


def compute_shap_native(cache, X):
    # API nativa de XGBoost — misma lógica que 07_post_iter1.ipynb
    feat_cols = cache['feature_cols']
    # usar el estimador base del primer fold de calibración
    booster = cache['model'].calibrated_classifiers_[0].estimator.get_booster()

    if isinstance(X, pd.DataFrame):
        X_arr = X[feat_cols].values
    else:
        X_arr = np.asarray(X)

    dm       = xgb.DMatrix(X_arr, feature_names=feat_cols)
    contribs = booster.predict(dm, pred_contribs=True)
    # XGBoost 3.x multiclass: (n, n_classes, n_features+1) — bias en última pos. del eje de features
    # XGBoost binary:         (n, n_features+1)
    if contribs.ndim == 3:
        shap_vals = contribs[:, :, :-1]   # (n, n_classes, n_features)
        bias      = contribs[:, :, -1]
    else:
        shap_vals = contribs[:, :-1]      # (n, n_features)
        bias      = contribs[:, -1]
    return shap_vals, bias


def predict(X, cache, compute_shap=False):
    model     = cache['model']
    feat_cols = cache['feature_cols']

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=feat_cols)
    else:
        X = X[feat_cols].copy()

    proba = model.predict_proba(X)       # shape (n, 3): SELL / HOLD / BUY
    pred  = proba.argmax(axis=1)

    shap_vals, _ = compute_shap_native(cache, X) if compute_shap else (None, None)

    results = []
    for i in range(len(X)):
        row = {
            'signal'  : SIGNAL_MAP[pred[i]],
            'position': POSITION_MAP[pred[i]],
            'p_sell'  : float(proba[i, 0]),
            'p_hold'  : float(proba[i, 1]),
            'p_buy'   : float(proba[i, 2]),
            'shap_values': None,
        }
        if compute_shap:
            cls_idx = pred[i]
            sv = shap_vals[i]
            # multiclase: sv shape (n_classes, n_features) → tomar fila de la clase predicha
            if sv.ndim == 2:
                sv_cls = sv[cls_idx, :]
            else:
                sv_cls = sv
            row['shap_values'] = dict(zip(feat_cols, sv_cls.tolist()))
        results.append(row)

    return results if len(results) > 1 else results[0]


def predict_single(ticker, date, feature_row, cache, compute_shap=False):
    result = predict(pd.DataFrame([feature_row]), cache, compute_shap=compute_shap)
    result['ticker'] = ticker
    result['date']   = str(date)
    return result


def top_shap_features(shap_dict, n=10):
    return sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:n]
