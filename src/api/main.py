# main.py — API FastAPI para el modelo de recomendación S&P 500
# Ejecutar: uvicorn src.api.main:app --reload
#
# Endpoints:
#   GET  /health
#   GET  /model_info
#   POST /predict        — un ticker → features + señal + probabilidades + SHAP
#   POST /predict_batch  — varios tickers → igual sin SHAP (más rápido)

import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.etl.feature_engine import build_features, get_model_inputs, TICKERS
from src.models.predictor import load_model, predict, top_shap_features

# ── configuración ──────────────────────────────────────────────────────────────
MODEL_PATH     = Path(os.getenv('MODEL_PATH',     'models/final/xgboost_3class_main.pkl'))
BLOOMBERG_FILE = os.getenv('BLOOMBERG_FILE', None)   # None → opt_* quedan NaN (NVDA)
WARMUP_MONTHS  = 14   # meses de historia para que los indicadores estén estabilizados

_cache = {}   # modelo cargado una sola vez en startup


@asynccontextmanager
async def lifespan(app: FastAPI):
    _cache['model'] = load_model(MODEL_PATH)
    print(f'API lista | modelo: {MODEL_PATH}')
    yield
    _cache.clear()


app = FastAPI(
    title='S&P 500 Recommendation API',
    description='XGBoost 3-Class ±1.5% — Grupo 20 MIAD Uniandes',
    version='1.0.0',
    lifespan=lifespan,
)


# ── schemas ────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    ticker: str
    shap: bool = False


class BatchRequest(BaseModel):
    tickers: list[str]


# ── helpers ────────────────────────────────────────────────────────────────────

def _fetch_latest_row(ticker):
    end   = datetime.today().strftime('%Y-%m-%d')
    start = (datetime.today() - timedelta(days=30 * WARMUP_MONTHS)).strftime('%Y-%m-%d')

    master = build_features(
        tickers=[ticker],
        start=start,
        end=end,
        bloomberg_file=BLOOMBERG_FILE,
    )
    inputs = get_model_inputs(master)
    if inputs.empty:
        raise HTTPException(status_code=404, detail=f'Sin datos para {ticker}')

    # última fila disponible
    last = inputs.sort_values('Date').iloc[[-1]]
    return last, str(last['Date'].iloc[0].date())


# ── endpoints ──────────────────────────────────────────────────────────────────

@app.get('/health')
def health():
    return {
        'status' : 'ok',
        'model'  : MODEL_PATH.name,
        'tickers': TICKERS,
    }


@app.get('/model_info')
def model_info():
    m = _cache['model']
    return {
        'model_type'   : m['model_type'],
        'target_col'   : m['target_col'],
        'feature_cols' : m['feature_cols'],
        'best_params'  : m['best_params'],
        'signals'      : {'0': 'SELL', '1': 'HOLD', '2': 'BUY'},
        'positions'    : {'SELL': -1, 'HOLD': 0, 'BUY': +1},
    }


@app.post('/predict')
def predict_ticker(req: PredictRequest):
    ticker = req.ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(status_code=400, detail=f'Ticker no soportado. Válidos: {TICKERS}')

    X, date = _fetch_latest_row(ticker)
    result  = predict(X, _cache['model'], compute_shap=req.shap)

    response = {
        'ticker'  : ticker,
        'date'    : date,
        'signal'  : result['signal'],
        'position': result['position'],
        'p_sell'  : result['p_sell'],
        'p_hold'  : result['p_hold'],
        'p_buy'   : result['p_buy'],
    }
    if req.shap and result['shap_values']:
        response['top_shap'] = dict(top_shap_features(result['shap_values'], n=10))

    return response


@app.post('/predict_batch')
def predict_batch(req: BatchRequest):
    tickers = [t.upper() for t in req.tickers]
    invalid = [t for t in tickers if t not in TICKERS]
    if invalid:
        raise HTTPException(status_code=400, detail=f'Tickers no soportados: {invalid}')

    end   = datetime.today().strftime('%Y-%m-%d')
    start = (datetime.today() - timedelta(days=30 * WARMUP_MONTHS)).strftime('%Y-%m-%d')

    master = build_features(
        tickers=tickers,
        start=start,
        end=end,
        bloomberg_file=BLOOMBERG_FILE,
    )
    inputs = get_model_inputs(master)

    # última fila por ticker
    last_rows = (inputs.sort_values('Date')
                       .groupby('Ticker')
                       .tail(1)
                       .reset_index(drop=True))

    feat_cols = _cache['model']['feature_cols']
    results   = predict(last_rows[feat_cols], _cache['model'], compute_shap=False)
    if not isinstance(results, list):
        results = [results]

    out = []
    for i, row in last_rows.iterrows():
        r = results[i] if len(results) > 1 else results[0]
        out.append({
            'ticker'  : row['Ticker'],
            'date'    : str(row['Date'].date()),
            'signal'  : r['signal'],
            'position': r['position'],
            'p_sell'  : r['p_sell'],
            'p_hold'  : r['p_hold'],
            'p_buy'   : r['p_buy'],
        })

    return {'predictions': out, 'n': len(out)}
