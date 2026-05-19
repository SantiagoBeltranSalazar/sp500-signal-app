# app.py — Dashboard Dash · S&P 500 Signal Model
# Ejecutar: uv run python src/dashboard/app.py

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
from dash.exceptions import PreventUpdate
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.trend import MACD as MACDIndicator
from ta.trend import SMAIndicator
from ta.volatility import BollingerBands

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.predictor import load_model, predict, top_shap_features

# ── rutas ─────────────────────────────────────────────────────────────────────

ROOT           = Path(__file__).resolve().parents[2]
MODEL_PATH     = ROOT / 'models/final/xgboost_3class_main.pkl'
MODEL_BIN_PATH = ROOT / 'models/final/xgboost_binary_alternative.pkl'
DATA_FEAT      = ROOT / 'data/features'
DATA_OHLCV     = ROOT / 'data/ohlcv'

TICKERS   = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM', 'UNH', 'XOM']
N_DISPLAY = 90    # días de trading visibles en los gráficos de indicadores
N_WARMUP  = 120   # días extra de warmup antes del display window

# ── paleta ────────────────────────────────────────────────────────────────────

NAVY   = '#0f172a'
NAVY3  = '#334155'
SLATE1 = '#f1f5f9'
SLATE2 = '#e2e8f0'
SLATE3 = '#cbd5e1'
SLATE4 = '#94a3b8'
SLATE5 = '#64748b'
ACCENT = '#2563eb'
BUY_C  = '#16a34a'
SELL_C = '#dc2626'
HOLD_C = '#d97706'
BIN_C  = '#7c3aed'

SIGNAL_COLOR  = {'BUY': BUY_C,     'SELL': SELL_C,   'HOLD': HOLD_C}
SIGNAL_BG     = {'BUY': '#f0fdf4', 'SELL': '#fef2f2', 'HOLD': '#fffbeb'}
SIGNAL_BORDER = {'BUY': '#bbf7d0', 'SELL': '#fecaca', 'HOLD': '#fde68a'}
SIGNAL_DESC   = {
    'BUY' : 'El modelo estima una subida superior al 1.5% en los próximos 10 días hábiles. Señal: posición larga.',
    'HOLD': 'El modelo no detecta tendencia clara. Movimiento esperado entre −1.5% y +1.5%. Sin operación recomendada.',
    'SELL': 'El modelo estima una caída superior al 1.5% en los próximos 10 días hábiles. Señal: posición corta.',
}

SHAP_LABELS = {
    'RSI_14'               : 'Índice de fuerza relativa',
    'OBV'                  : 'Flujo de volumen acumulado',
    'VIX_SMA_5'            : 'VIX promedio 5 días',
    'Spread_Future_Spot'   : 'Spread Futuro S&P 500 vs Spot',
    'log_return_lag1'      : 'Retorno del día anterior',
    'log_return_lag2'      : 'Retorno de hace 2 días',
    'log_return_lag5'      : 'Retorno de hace 5 días',
    'opt_ivol_delta'       : 'Volatilidad implícita (delta)',
    'MACD_signal_rel'      : 'Señal MACD normalizada',
    'opt_put_call_vol_ratio': 'Ratio volumen Put/Call',
    'VIX_MACD_Signal'      : 'Señal MACD del VIX',
    'BB_lower_rel'         : 'Banda Bollinger inferior rel.',
    'BB_width'             : 'Ancho Bandas de Bollinger',
    'MACD_diff_rel'        : 'Diferencial MACD normalizado',
    'EMA_12_rel'           : 'EMA 12 días normalizada',
    'ATR_14_rel'           : 'Rango verdadero medio norm.',
    'Volatility_10'        : 'Volatilidad histórica 10 días',
    'Volume_change'        : 'Cambio de volumen diario',
    'Volume'               : 'Volumen futuros ES=F',
    'Volume_Change'        : 'Cambio volumen futuros',
    'opt_ivol_vs_vix'      : 'IV opciones vs VIX',
    'opt_ivol_vs_hvol'     : 'IV opciones vs vol. histórica',
    'opt_hist_vol_30d'     : 'Volatilidad histórica 30 días',
    'opt_put_call_oi_ratio': 'Ratio OI Put/Call',
    'opt_impl_vol_call'    : 'Vol. implícita calls',
    'Future_RSI_14'        : 'RSI futuros ES=F',
    'Future_Returns'       : 'Retorno diario futuros',
    'Future_Close'         : 'Precio cierre futuros ES=F',
    'Future_SMA_20'        : 'SMA 20 días futuros',
    'Future_Volatility_10' : 'Volatilidad 10 días futuros',
}

SHAP_DESC = {
    'RSI_14'               : 'Índice de Fuerza Relativa (14 días). Mide si la acción está sobrecomprada (>70) o sobrevendida (<30). Valores extremos suelen preceder correcciones.',
    'OBV'                  : 'On-Balance Volume. Acumula el volumen según la dirección del precio. Una tendencia creciente indica presión compradora sostenida.',
    'VIX_SMA_5'            : 'Media móvil de 5 días del VIX. El VIX mide el "miedo" del mercado: por encima de 20 indica alta incertidumbre, por encima de 30 es pánico.',
    'Spread_Future_Spot'   : 'Diferencia entre el precio del futuro del S&P 500 (ES=F) y el precio spot. Un spread positivo indica expectativas alcistas a corto plazo.',
    'log_return_lag1'      : 'Retorno logarítmico del día hábil anterior. Captura el momentum más reciente de la acción.',
    'log_return_lag2'      : 'Retorno logarítmico de hace 2 días hábiles. Junto con lag1 y lag5 permite detectar patrones de momentum.',
    'log_return_lag5'      : 'Retorno logarítmico de hace 5 días hábiles (aproximadamente 1 semana).',
    'opt_ivol_delta'       : 'Volatilidad implícita ponderada por delta. Refleja las expectativas de movimiento que el mercado de opciones descuenta.',
    'MACD_signal_rel'      : 'Línea de señal del MACD normalizada por el precio. Detecta cruces que anticipan cambios de tendencia.',
    'opt_put_call_vol_ratio': 'Ratio de volumen Put/Call. Un valor alto indica más compra de protección bajista que alcista — sentimiento de cobertura.',
    'VIX_MACD_Signal'      : 'Señal MACD calculada sobre el VIX. Detecta aceleraciones o desaceleraciones en la volatilidad del mercado.',
    'BB_lower_rel'         : 'Distancia relativa al precio de la banda de Bollinger inferior. Valores negativos indican que el precio rompió el soporte estadístico.',
    'BB_width'             : 'Ancho de las Bandas de Bollinger. Un ancho alto indica alta volatilidad reciente; uno bajo, compresión antes de un movimiento.',
    'MACD_diff_rel'        : 'Histograma MACD normalizado. Muestra la velocidad y dirección del cambio de momentum.',
    'EMA_12_rel'           : 'Posición del precio relativa a la media exponencial de 12 días. Indica tendencia de corto plazo.',
    'ATR_14_rel'           : 'Average True Range de 14 días normalizado. Mide la volatilidad de precio de corto plazo.',
    'Volatility_10'        : 'Desviación estándar de los retornos logarítmicos de los últimos 10 días. Volatilidad histórica reciente.',
    'Volume_change'        : 'Cambio porcentual del volumen de la acción respecto al día anterior.',
    'Volume'               : 'Volumen operado en el contrato de futuros ES=F ese día.',
    'Volume_Change'        : 'Cambio del volumen en el contrato de futuros ES=F.',
    'opt_ivol_vs_vix'      : 'Ratio entre la volatilidad implícita de las opciones de la acción y el VIX. Detecta si la acción cotiza más cara o barata que el mercado.',
    'opt_ivol_vs_hvol'     : 'Ratio entre volatilidad implícita y volatilidad histórica. Mayor a 1 indica que el mercado espera más movimiento del observado recientemente.',
    'opt_hist_vol_30d'     : 'Volatilidad histórica de 30 días calculada a partir de precios de opciones (Bloomberg).',
    'opt_put_call_oi_ratio': 'Ratio de open interest Put/Call. Indica el posicionamiento estructural acumulado en opciones.',
    'opt_impl_vol_call'    : 'Volatilidad implícita de las opciones call ATM. Refleja el costo de cobertura alcista en el mercado de opciones.',
    'Future_RSI_14'        : 'RSI de 14 días calculado sobre el precio del futuro ES=F.',
    'Future_Returns'       : 'Retorno logarítmico diario del contrato de futuros ES=F.',
    'Future_Close'         : 'Precio de cierre del contrato de futuros S&P 500 (ES=F).',
    'Future_SMA_20'        : 'Media móvil simple de 20 días del futuro ES=F.',
    'Future_Volatility_10' : 'Volatilidad de 10 días del contrato de futuros ES=F.',
}

FUND_FIELDS = [
    ('ROE',                 'Retorno sobre patrimonio (ROE)',   '{:.1%}'),
    ('ROA',                 'Retorno sobre activos (ROA)',      '{:.1%}'),
    ('Profit_Margin',       'Margen de beneficio neto',         '{:.1%}'),
    ('Operating_Margin',    'Margen operativo',                 '{:.1%}'),
    ('Gross_Margin',        'Margen bruto',                     '{:.1%}'),
    ('EPS',                 'Beneficio por acción (EPS)',       '${:.2f}'),
    ('P_E_Ratio',           'Ratio precio / beneficio (P/E)',   '{:.1f}x'),
    ('P_B_Ratio',           'Ratio precio / valor libro (P/B)', '{:.2f}x'),
    ('Debt_to_Equity',      'Deuda sobre patrimonio',           '{:.1f}x'),
    ('Current_Ratio',       'Ratio corriente',                  '{:.2f}'),
    ('Quick_Ratio',         'Ratio rápido',                     '{:.2f}'),
    ('Revenue_Growth_YoY',  'Crecimiento ingresos (YoY)',       '{:.1%}'),
    ('Earnings_Growth_YoY', 'Crecimiento ganancias (YoY)',      '{:.1%}'),
]

PLOTLY_LAYOUT = dict(
    paper_bgcolor='white',
    plot_bgcolor ='white',
    font         =dict(family='IBM Plex Sans, system-ui, sans-serif', size=11, color=NAVY),
    margin       =dict(l=40, r=12, t=28, b=32),
    xaxis        =dict(showgrid=True, gridcolor=SLATE2, zeroline=False, linecolor=SLATE3),
    yaxis        =dict(showgrid=True, gridcolor=SLATE2, zeroline=False, linecolor=SLATE3),
)

# ── carga de datos ────────────────────────────────────────────────────────────

print('Cargando modelos...')
_cache     = load_model(MODEL_PATH)
_cache_bin = load_model(MODEL_BIN_PATH)
FEAT       = _cache['feature_cols']
FEAT_BIN   = _cache_bin['feature_cols']

print('Cargando parquets...')
df_test = pd.read_parquet(DATA_FEAT / 'test.parquet')
df_oos  = pd.read_parquet(DATA_FEAT / 'master_features_oos.parquet')
df_test['Date'] = pd.to_datetime(df_test['Date']).dt.normalize()
df_oos['Date']  = pd.to_datetime(df_oos['Date']).dt.normalize()

ohlcv = {}
for _t in TICKERS + ['SPY']:
    _df = pd.read_parquet(DATA_OHLCV / f'{_t}_ohlcv.parquet')
    _df.index = pd.to_datetime(_df.index).normalize()
    ohlcv[_t] = _df

# ── equity curves (pre-computadas al arrancar) ────────────────────────────────

def _sharpe(s):
    return float(s.mean() / s.std() * np.sqrt(252)) if s.std() > 0 else 0.0

def _build_equity(df_feats, feat_cols, start, end, spy_ohlcv, model, is_binary=False):
    subset = df_feats[
        (df_feats['Date'] >= start) &
        (df_feats['Date'] <= end) &
        (df_feats['log_return_next'].notna())
    ].copy().sort_values(['Ticker', 'Date']).reset_index(drop=True)

    # usar predict_proba directamente — funciona para 2 o 3 clases
    proba = model.predict_proba(subset[feat_cols])
    pred  = proba.argmax(axis=1)
    if is_binary:
        pos_map = {0: 0.0, 1: 1.0}          # 0=flat, 1=long
    else:
        pos_map = {0: -1.0, 1: 0.0, 2: 1.0} # 0=SELL, 1=HOLD, 2=BUY
    subset['position'] = [pos_map[p] for p in pred]
    subset['strat_r']  = subset['position'] * subset['log_return_next']

    strat_avg = subset.groupby('Date')['strat_r'].mean()
    bh_avg    = subset.groupby('Date')['log_return_next'].mean()

    spy_r = np.log1p(spy_ohlcv['Close'].pct_change()).dropna()
    spy_r = spy_r[
        (spy_r.index >= pd.Timestamp(start)) &
        (spy_r.index <= pd.Timestamp(end))
    ].reindex(strat_avg.index).fillna(0)

    eq_m   = np.exp(strat_avg.cumsum())
    eq_bh  = np.exp(bh_avg.cumsum())
    eq_spy = np.exp(spy_r.cumsum())

    n_beat = 0
    for t in TICKERS:
        t_sub = subset[subset['Ticker'] == t]
        if not t_sub.empty and _sharpe(t_sub['strat_r']) > _sharpe(t_sub['log_return_next']):
            n_beat += 1

    return {
        'dates'      : strat_avg.index.strftime('%Y-%m-%d').tolist(),
        'modelo'     : eq_m.tolist(),
        'bh'         : eq_bh.tolist(),
        'spy'        : eq_spy.tolist(),
        'sharpe'     : _sharpe(strat_avg),
        'sharpe_bh'  : _sharpe(bh_avg),
        'sharpe_spy' : _sharpe(spy_r),
        'retorno'    : float(eq_m.iloc[-1] - 1),
        'retorno_bh' : float(eq_bh.iloc[-1] - 1),
        'retorno_spy': float(eq_spy.iloc[-1] - 1),
        'maxdd'      : float(((eq_m - eq_m.cummax()) / eq_m.cummax()).min()),
        'n_beat'     : n_beat,
    }

print('Computando equity curves...')
TEST_EQ = _build_equity(df_test, FEAT,     '2023-01-01', '2024-12-31', ohlcv['SPY'], _cache['model'])
OOS_EQ  = _build_equity(df_oos,  FEAT,     '2025-01-01', '2025-12-31', ohlcv['SPY'], _cache['model'])
OOS_BIN = _build_equity(df_oos,  FEAT_BIN, '2025-01-01', '2025-12-31', ohlcv['SPY'], _cache_bin['model'], is_binary=True)
print(f'Test  | sharpe={TEST_EQ["sharpe"]:.3f} | ret={TEST_EQ["retorno"]:.2%} | maxdd={TEST_EQ["maxdd"]:.2%}')
print(f'OOS   | sharpe={OOS_EQ["sharpe"]:.3f}  | ret={OOS_EQ["retorno"]:.2%}  | maxdd={OOS_EQ["maxdd"]:.2%}')
print(f'Bin   | sharpe={OOS_BIN["sharpe"]:.3f} | ret={OOS_BIN["retorno"]:.2%} | maxdd={OOS_BIN["maxdd"]:.2%}')
print('Dashboard listo.\n')

# ── valores iniciales ─────────────────────────────────────────────────────────

_INIT_TICKER = 'AAPL'
_init_dates  = df_test[df_test['Ticker'] == _INIT_TICKER]['Date'].sort_values()
_INIT_MIN    = _init_dates.iloc[0].strftime('%Y-%m-%d')
_INIT_MAX    = _init_dates.iloc[-1].strftime('%Y-%m-%d')
_INIT_DATE   = _init_dates.iloc[-11].strftime('%Y-%m-%d')

# ── helpers de datos ──────────────────────────────────────────────────────────

def _nearest_date(ticker, date_str):
    avail  = df_test[df_test['Ticker'] == ticker]['Date'].sort_values().reset_index(drop=True)
    target = pd.Timestamp(date_str)
    return avail.iloc[(avail - target).abs().argmin()]

def _lookup_row(ticker, date_str):
    date = _nearest_date(ticker, date_str)
    mask = (df_test['Ticker'] == ticker) & (df_test['Date'] == date)
    return df_test[mask].iloc[0], date

def _prep_indicators_df(ticker, date):
    df  = ohlcv[ticker]
    cd  = pd.Timestamp(date)
    pre = df.index[df.index <= cd]
    fut = df.index[df.index > cd]

    n_load = N_DISPLAY + N_WARMUP
    start  = pre[-n_load] if len(pre) >= n_load else pre[0]
    end    = fut[1]       if len(fut) >= 2 else (fut[-1] if len(fut) > 0 else cd)
    df_w   = df[(df.index >= start) & (df.index <= end)].copy()

    close        = df_w['Close']
    df_w['SMA_20'] = SMAIndicator(close, 20).sma_indicator()
    df_w['SMA_50'] = SMAIndicator(close, 50).sma_indicator()
    df_w['EMA_12'] = EMAIndicator(close, 12).ema_indicator()
    df_w['RSI']    = RSIIndicator(close, 14).rsi()
    macd_ind       = MACDIndicator(close, 26, 12, 9)
    df_w['MACD']   = macd_ind.macd()
    df_w['MACDS']  = macd_ind.macd_signal()
    df_w['MACDD']  = macd_ind.macd_diff()
    bb             = BollingerBands(close, 20, 2)
    df_w['BB_u']   = bb.bollinger_hband()
    df_w['BB_m']   = bb.bollinger_mavg()
    df_w['BB_l']   = bb.bollinger_lband()

    # recortar al display window
    pre_d = df_w.index[df_w.index <= cd]
    pos_d = df_w.index[df_w.index > cd]
    s_d   = pre_d[-N_DISPLAY] if len(pre_d) >= N_DISPLAY else pre_d[0]
    e_d   = pos_d[1] if len(pos_d) >= 2 else (pos_d[-1] if len(pos_d) > 0 else cd)
    return df_w[(df_w.index >= s_d) & (df_w.index <= e_d)]

# ── figuras Plotly ────────────────────────────────────────────────────────────

def _fig_empty(h=220):
    fig = go.Figure()
    fig.update_layout(**PLOTLY_LAYOUT, height=h)
    return fig

def fig_candlestick_context(df, date):
    cd  = pd.Timestamp(date)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name='Precio',
        increasing_line_color=BUY_C, decreasing_line_color=SELL_C,
        increasing_fillcolor='rgba(22,163,74,0.3)',
        decreasing_fillcolor='rgba(220,38,38,0.3)',
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20',
                             line=dict(color=ACCENT, width=1, dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50',
                             line=dict(color=HOLD_C, width=1, dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_12'], name='EMA 12',
                             line=dict(color=BUY_C, width=1, dash='dash')))
    if cd in df.index:
        fig.add_vline(x=cd, line=dict(color=NAVY3, width=1.5, dash='dash'))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title=f'Precio + Medias Móviles ({N_DISPLAY} días)',
                      legend=dict(orientation='h', y=1.08, x=0, font_size=10),
                      xaxis_rangeslider_visible=False)
    return fig

def fig_rsi(df, date):
    cd     = pd.Timestamp(date)
    layout = {**PLOTLY_LAYOUT, 'yaxis': dict(range=[0, 100], showgrid=True, gridcolor=SLATE2)}
    fig    = go.Figure()
    fig.add_hrect(y0=70, y1=100, fillcolor='#fee2e2', opacity=0.3, line_width=0)
    fig.add_hrect(y0=0,  y1=30,  fillcolor='#dcfce7', opacity=0.3, line_width=0)
    fig.add_hline(y=70, line=dict(color=SELL_C, width=1, dash='dot'))
    fig.add_hline(y=30, line=dict(color=BUY_C,  width=1, dash='dot'))
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI 14',
                             line=dict(color=ACCENT, width=1.5)))
    if cd in df.index:
        fig.add_vline(x=cd, line=dict(color=NAVY3, width=1.5, dash='dash'))
    fig.update_layout(**layout, title='RSI 14 — Índice de Fuerza Relativa')
    return fig

def fig_macd(df, date):
    cd     = pd.Timestamp(date)
    colors = [BUY_C if v >= 0 else SELL_C for v in df['MACDD'].fillna(0)]
    fig    = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df['MACDD'], name='Histograma',
                         marker_color=colors, opacity=0.6))
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'],  name='MACD',
                             line=dict(color=ACCENT, width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df['MACDS'], name='Señal',
                             line=dict(color=SELL_C, width=1.5)))
    if cd in df.index:
        fig.add_vline(x=cd, line=dict(color=NAVY3, width=1.5, dash='dash'))
    fig.update_layout(**PLOTLY_LAYOUT, title='MACD + Señal + Histograma',
                      legend=dict(orientation='h', y=1.08, x=0, font_size=10),
                      barmode='overlay')
    return fig

def fig_bollinger(df, date):
    cd  = pd.Timestamp(date)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(df.index) + list(df.index[::-1]),
        y=list(df['BB_u']) + list(df['BB_l'][::-1]),
        fill='toself', fillcolor='rgba(37,99,235,0.05)',
        line=dict(color='rgba(0,0,0,0)'), showlegend=False,
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_u'], name='Superior',
                             line=dict(color=ACCENT, width=1, dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_l'], name='Inferior',
                             line=dict(color=ACCENT, width=1, dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_m'], name='Media',
                             line=dict(color=SLATE4, width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Precio',
                             line=dict(color=NAVY, width=1.5)))
    if cd in df.index:
        fig.add_vline(x=cd, line=dict(color=NAVY3, width=1.5, dash='dash'))
    fig.update_layout(**PLOTLY_LAYOUT, title='Bandas de Bollinger',
                      legend=dict(orientation='h', y=1.08, x=0, font_size=10))
    return fig

def fig_verification(ticker, date, signal):
    df  = ohlcv[ticker]
    cd  = pd.Timestamp(date)
    pre = df.index[df.index <= cd]
    fut = df.index[df.index > cd]
    if len(pre) == 0 or len(fut) == 0:
        return _fig_empty(240)
    all_idx = pre[-1:].union(fut[:10])
    df_plot = df.loc[all_idx]
    color   = SIGNAL_COLOR.get(signal, SLATE4)
    fig     = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_plot.index, open=df_plot['Open'], high=df_plot['High'],
        low=df_plot['Low'], close=df_plot['Close'], name='Precio',
        increasing_line_color=BUY_C, decreasing_line_color=SELL_C,
        increasing_fillcolor='rgba(22,163,74,0.3)',
        decreasing_fillcolor='rgba(220,38,38,0.3)',
    ))
    fig.add_vline(x=pre[-1], line=dict(color=color, width=2, dash='dash'))
    fig.add_annotation(x=pre[-1], y=float(df_plot['Close'].iloc[0]),
                       text=f'  {signal}', showarrow=False,
                       font=dict(size=11, color=color, family='IBM Plex Mono'),
                       xanchor='left')
    if len(df_plot) >= 2:
        ret = float(df_plot['Close'].iloc[-1] / df_plot['Close'].iloc[0] - 1)
        rc  = BUY_C if ret > 0.015 else (SELL_C if ret < -0.015 else HOLD_C)
        fig.add_annotation(x=df_plot.index[-1], y=float(df_plot['Close'].iloc[-1]),
                           text=f'  {ret:+.2%}', showarrow=False,
                           font=dict(size=11, color=rc, family='IBM Plex Mono'),
                           xanchor='left')
    fig.update_layout(**PLOTLY_LAYOUT,
                      title='Verificación: los 10 días hábiles posteriores a la señal',
                      xaxis_rangeslider_visible=False)
    return fig

def fig_vix_context(date, df_ind):
    cd   = pd.Timestamp(date)
    df_r = df_test[df_test['Ticker'] == 'AAPL'].sort_values('Date')
    df_w = df_r[(df_r['Date'] >= df_ind.index[0]) & (df_r['Date'] <= cd)]
    fig  = go.Figure()
    fig.add_hline(y=20, line=dict(color=HOLD_C, width=1, dash='dot'))
    fig.add_hline(y=30, line=dict(color=SELL_C, width=1, dash='dot'))
    fig.add_trace(go.Scatter(x=df_w['Date'], y=df_w['VIX'], name='VIX',
                             fill='tozeroy', fillcolor='rgba(220,38,38,0.08)',
                             line=dict(color=SELL_C, width=1.5)))
    fig.add_vline(x=cd, line=dict(color=NAVY3, width=1.5, dash='dash'))
    fig.update_layout(**PLOTLY_LAYOUT, title='VIX — Índice de Volatilidad del Mercado')
    return fig

def fig_futures_context(date, df_ind):
    cd   = pd.Timestamp(date)
    df_r = df_test[df_test['Ticker'] == 'AAPL'].sort_values('Date')
    df_w = df_r[(df_r['Date'] >= df_ind.index[0]) & (df_r['Date'] <= cd)]
    fig  = go.Figure()
    fig.add_trace(go.Scatter(x=df_w['Date'], y=df_w['Future_Close'], name='ES=F',
                             line=dict(color=ACCENT, width=1.5)))
    fig.add_vline(x=cd, line=dict(color=NAVY3, width=1.5, dash='dash'))
    fig.update_layout(**PLOTLY_LAYOUT, title='Futuros S&P 500 (ES=F) — precio de cierre')
    return fig

def fig_equity(eq_main, eq_bin=None, title=''):
    if not eq_main:
        return _fig_empty(260)
    dates = pd.to_datetime(eq_main['dates'])
    fig   = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=eq_main['modelo'], name='Modelo 3-Class',
                             line=dict(color=ACCENT, width=2)))
    if eq_bin:
        fig.add_trace(go.Scatter(x=pd.to_datetime(eq_bin['dates']), y=eq_bin['modelo'],
                                 name='Modelo Binario',
                                 line=dict(color=BIN_C, width=1.8, dash='dashdot')))
    fig.add_trace(go.Scatter(x=dates, y=eq_main['bh'],  name='B&H 10 acciones',
                             line=dict(color=NAVY3, width=1.5, dash='dot')))
    fig.add_trace(go.Scatter(x=dates, y=eq_main['spy'], name='SPY (S&P 500)',
                             line=dict(color=SLATE4, width=1.5, dash='dash')))
    fig.add_hline(y=1, line=dict(color=SLATE3, width=1))
    layout = {**PLOTLY_LAYOUT, 'yaxis': dict(tickformat='.2f', showgrid=True, gridcolor=SLATE2)}
    fig.update_layout(**layout, title=title,
                      legend=dict(orientation='h', y=1.08, x=0, font_size=11))
    return fig

# ── tabla SHAP (HTML, con tooltip por variable) ───────────────────────────────

def shap_table(shap_top):
    if not shap_top:
        return html.Div('Sin datos SHAP disponibles.',
                        style={'padding': '20px', 'color': SLATE4, 'fontSize': '12px'})
    rows = []
    for item in shap_top:
        feat, val = item[0], item[1]
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        label = SHAP_LABELS.get(feat, feat)
        desc  = SHAP_DESC.get(feat, '')
        color = ACCENT if val >= 0 else SELL_C
        bg    = 'rgba(37,99,235,0.06)' if val >= 0 else 'rgba(220,38,38,0.06)'
        sign  = f'+{val:.3f}' if val >= 0 else f'{val:.3f}'

        rows.append(html.Div([
            # dot de color
            html.Div(style={
                'width': '8px', 'height': '8px', 'borderRadius': '50%',
                'background': color, 'flexShrink': '0', 'marginTop': '2px',
            }),
            # nombre + código técnico
            html.Div([
                html.Div(label, style={'fontSize': '12px', 'fontWeight': '500', 'color': NAVY}),
                html.Div(feat,  style={'fontSize': '10px', 'color': SLATE4, 'fontFamily': 'IBM Plex Mono, monospace'}),
            ], style={'flex': '1', 'minWidth': '0'}),
            # valor
            html.Div(sign, style={
                'fontSize': '12px', 'fontFamily': 'IBM Plex Mono, monospace',
                'fontWeight': '500', 'color': color, 'flexShrink': '0',
            }),
            # tooltip (CSS puro — aparece al hover del padre)
            html.Div(desc, className='shap-tip') if desc else html.Span(),
        ], className='shap-row', style={
            'display': 'flex', 'alignItems': 'flex-start', 'gap': '10px',
            'padding': '7px 12px', 'borderBottom': f'1px solid {SLATE2}',
            'background': bg, 'position': 'relative', 'cursor': 'default',
        }))

    return html.Div(rows)

# ── layout helpers ────────────────────────────────────────────────────────────

def badge(signal):
    c  = SIGNAL_COLOR.get(signal, SLATE4)
    bg = SIGNAL_BG.get(signal, SLATE1)
    bd = SIGNAL_BORDER.get(signal, SLATE2)
    return html.Div(signal, style={
        'display': 'inline-block', 'padding': '6px 22px',
        'background': bg, 'color': c, 'border': f'1.5px solid {bd}',
        'fontSize': '22px', 'fontWeight': '600',
        'letterSpacing': '0.08em', 'fontFamily': 'IBM Plex Mono, monospace',
    })

def prob_bar(label, pct, color):
    return html.Div([
        html.Div([
            html.Span(label, style={'color': SLATE5, 'fontSize': '11.5px',
                                    'width': '36px', 'display': 'inline-block'}),
            html.Div(style={
                'display': 'inline-block', 'verticalAlign': 'middle',
                'width': f'{pct*100:.0f}%', 'maxWidth': '55%', 'minWidth': '4px',
                'height': '6px', 'background': color, 'borderRadius': '2px',
                'margin': '0 8px',
            }),
            html.Span(f'{pct*100:.0f}%', style={
                'color': NAVY, 'fontSize': '12px',
                'fontFamily': 'IBM Plex Mono, monospace', 'fontWeight': '500',
            }),
        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '6px'}),
    ])

def info_icon(text):
    return html.Span([
        html.Span('ⓘ', style={'color': SLATE4, 'fontSize': '12px',
                               'marginLeft': '4px', 'cursor': 'help'}),
        html.Div(text, className='info-tip'),
    ], className='info-wrap')

def metric_card(label, value, delta=None, delta2=None):
    return html.Div([
        html.Div(label, style={'fontSize': '11px', 'color': SLATE5,
                               'textTransform': 'uppercase', 'letterSpacing': '0.05em',
                               'marginBottom': '6px'}),
        html.Div(value, style={'fontSize': '22px', 'fontWeight': '600',
                               'fontFamily': 'IBM Plex Mono, monospace', 'color': NAVY}),
        html.Div(delta,  style={'fontSize': '11px', 'color': SLATE5, 'marginTop': '4px'})
        if delta  else html.Div(),
        html.Div(delta2, style={'fontSize': '11px', 'color': SLATE4, 'marginTop': '2px'})
        if delta2 else html.Div(),
    ], style={
        'background': 'white', 'border': f'1px solid {SLATE2}',
        'padding': '16px 18px', 'flex': '1',
    })

def section_header(title, subtitle=None):
    return html.Div([
        html.Div(title, style={
            'fontSize': '10.5px', 'color': SLATE5,
            'textTransform': 'uppercase', 'letterSpacing': '0.07em', 'fontWeight': '500',
        }),
        html.Div(subtitle, style={'fontSize': '11px', 'color': SLATE4, 'marginTop': '4px'})
        if subtitle else html.Div(),
    ], style={'padding': '16px 20px 0'})

def tab_description(text):
    return html.Div(text, style={
        'fontSize': '11.5px', 'color': SLATE5, 'padding': '8px 20px',
        'background': SLATE1, 'borderBottom': f'1px solid {SLATE2}',
    })

def fund_table(rows):
    if not rows:
        return html.Div('Sin datos fundamentales.',
                        style={'padding': '20px', 'color': SLATE4, 'fontSize': '12px'})
    return html.Div([
        html.Table([
            html.Tbody([
                html.Tr([
                    html.Td(lbl, style={
                        'padding': '7px 16px', 'fontSize': '12px', 'color': SLATE5,
                        'borderBottom': f'1px solid {SLATE2}', 'whiteSpace': 'nowrap',
                    }),
                    html.Td(val, style={
                        'padding': '7px 16px', 'fontSize': '12px', 'fontWeight': '500',
                        'color': NAVY, 'borderBottom': f'1px solid {SLATE2}',
                        'fontFamily': 'IBM Plex Mono, monospace',
                    }),
                ]) for lbl, val in rows
            ])
        ], style={'borderCollapse': 'collapse', 'width': '100%'}),
        html.Div(
            'Nota: datos de contexto financiero. El modelo no los usa como variables de entrada.',
            style={'fontSize': '11px', 'color': SLATE4, 'padding': '10px 16px',
                   'borderTop': f'1px solid {SLATE2}', 'fontStyle': 'italic'},
        ),
    ])

def results_divider():
    return html.Div([
        html.Div(style={'flex': '1', 'height': '1px', 'background': SLATE2}),
        html.Div('RESULTADOS GLOBALES DEL MODELO', style={
            'fontSize': '10px', 'color': SLATE4, 'letterSpacing': '0.1em',
            'padding': '0 20px', 'whiteSpace': 'nowrap',
        }),
        html.Div(style={'flex': '1', 'height': '1px', 'background': SLATE2}),
    ], style={'display': 'flex', 'alignItems': 'center', 'margin': '16px 0 8px'})

# ── app ───────────────────────────────────────────────────────────────────────

app = Dash(__name__, suppress_callback_exceptions=True)

app.index_string = '''<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>S&P 500 · Signal Model</title>
    {%favicon%}
    {%css%}
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; }
        body { margin: 0; font-family: "IBM Plex Sans", system-ui, sans-serif;
               background: #f1f5f9; color: #0f172a; font-size: 13px; }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
        /* date picker — clases de la versión actual de Dash */
        .dash-datepicker { background: transparent !important; }
        .dash-datepicker-input-wrapper {
            background: white !important;
            border: 1px solid #cccccc !important;
            border-radius: 4px !important;
            padding: 0 8px !important;
            height: 34px !important;
            display: inline-flex !important;
            align-items: center !important;
            gap: 4px !important;
            cursor: pointer !important;
            box-sizing: border-box !important;
        }
        .dash-datepicker-input {
            background: white !important;
            color: #0f172a !important;
            font-size: 12px !important;
            font-family: "IBM Plex Sans", system-ui, sans-serif !important;
            border: none !important;
            outline: none !important;
            padding: 0 !important;
            height: 100% !important;
        }
        .dash-datepicker-caret-icon { color: #666 !important; }

        /* tooltip para filas SHAP */
        .shap-row:hover { background: rgba(37,99,235,0.04) !important; }
        .shap-tip {
            display: none;
            position: absolute;
            left: 0; bottom: 100%; top: auto;
            margin-bottom: 4px;
            background: #1e293b;
            color: #e2e8f0;
            font-size: 11px;
            line-height: 1.5;
            padding: 8px 12px;
            border-radius: 4px;
            width: 320px;
            z-index: 9999;
            pointer-events: none;
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        }
        .shap-row:hover .shap-tip { display: block; }

        /* tooltip para iconos ⓘ */
        .info-wrap {
            position: relative;
            display: inline-block;
        }
        .info-tip {
            display: none;
            position: absolute;
            left: 50%; transform: translateX(-50%);
            top: calc(100% + 4px);
            background: #1e293b;
            color: #e2e8f0;
            font-size: 11px;
            line-height: 1.5;
            padding: 8px 12px;
            border-radius: 4px;
            width: 260px;
            z-index: 9999;
            pointer-events: none;
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        }
        .info-wrap:hover .info-tip { display: block; }
    </style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>'''

_tab_active   = {'background': ACCENT, 'color': 'white', 'border': 'none',
                 'padding': '6px 16px', 'cursor': 'pointer', 'fontSize': '12px',
                 'fontFamily': 'inherit'}
_tab_inactive = {'background': 'white', 'color': NAVY3, 'border': f'1px solid {SLATE2}',
                 'padding': '6px 16px', 'cursor': 'pointer', 'fontSize': '12px',
                 'fontFamily': 'inherit'}

# ── layout ────────────────────────────────────────────────────────────────────

app.layout = html.Div([

    dcc.Store(id='store-result'),

    # HEADER
    html.Div([
        html.Div([
            html.Span('S&P 500', style={'color': 'white', 'fontWeight': '600',
                                        'fontSize': '15px', 'letterSpacing': '0.04em'}),
            html.Span(' · Signal Model', style={'color': SLATE4, 'fontSize': '14px'}),
        ]),
        html.Div([
            dcc.Dropdown(
                id='ticker-dropdown',
                options=[{'label': t, 'value': t} for t in TICKERS],
                value=_INIT_TICKER, clearable=False,
                style={'width': '110px', 'fontSize': '12px'},
            ),
            dcc.DatePickerSingle(
                id='date-picker',
                min_date_allowed=_INIT_MIN,
                max_date_allowed=_INIT_MAX,
                date=_INIT_DATE,
                display_format='DD MMM YYYY',
            ),
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'}),
        html.Div(id='header-status', children='—',
                 style={'color': SLATE4, 'fontSize': '11.5px', 'marginLeft': 'auto'}),
    ], style={
        'background': NAVY, 'padding': '0 24px', 'height': '52px',
        'display': 'flex', 'alignItems': 'center', 'gap': '24px',
        'borderBottom': f'1px solid {NAVY3}',
    }),

    # CUERPO
    html.Div([

        # Fila 1: señal + SHAP
        html.Div([
            # panel señal
            html.Div([
                html.Div('RECOMENDACIÓN', style={
                    'fontSize': '10.5px', 'color': SLATE5,
                    'textTransform': 'uppercase', 'letterSpacing': '0.07em',
                    'marginBottom': '14px', 'fontWeight': '500',
                }),
                html.Div(id='signal-badge', children=badge('—'), style={'marginBottom': '10px'}),
                html.Div(id='signal-desc', children='Selecciona un ticker y una fecha.',
                         style={'fontSize': '11.5px', 'color': SLATE5,
                                'lineHeight': '1.5', 'marginBottom': '10px'}),
                html.Div(id='signal-date',    children='Fecha: —',
                         style={'fontSize': '11px', 'color': SLATE4, 'marginBottom': '4px'}),
                html.Div(id='signal-horizon', children='',
                         style={'fontSize': '11px', 'color': SLATE4, 'marginBottom': '18px'}),
                html.Div('PROBABILIDADES', style={
                    'fontSize': '10px', 'color': SLATE5,
                    'textTransform': 'uppercase', 'letterSpacing': '0.06em',
                    'marginBottom': '10px',
                }),
                html.Div(id='prob-bars'),
            ], style={
                'background': 'white', 'border': f'1px solid {SLATE2}',
                'padding': '20px 22px', 'width': '230px', 'flexShrink': '0',
            }),

            # panel SHAP
            html.Div([
                html.Div([
                    html.Span('EXPLICACIÓN SHAP', style={
                        'fontSize': '10.5px', 'color': SLATE5,
                        'textTransform': 'uppercase', 'letterSpacing': '0.07em', 'fontWeight': '500',
                    }),
                    info_icon(
                        'SHAP (SHapley Additive exPlanations) explica por qué el modelo tomó esta '
                        'decisión para esta fecha específica, no en general. Cada fila es una '
                        'variable del mercado. Los valores positivos (azul) empujaron al modelo '
                        'hacia la señal emitida. Los negativos (rojo) la frenaron. Cuanto mayor '
                        'el número, más peso tuvo esa variable en esta predicción.'
                    ),
                ], style={'marginBottom': '6px'}),
                html.Div(
                    'Pasa el cursor sobre cada variable para ver qué mide. '
                    'Azul = empuja hacia la señal actual · Rojo = la modera.',
                    style={'fontSize': '11px', 'color': SLATE4, 'lineHeight': '1.5',
                           'marginBottom': '10px'}
                ),
                dcc.Loading(
                    html.Div(id='shap-table'),
                    type='dot', color=ACCENT,
                ),
            ], style={
                'background': 'white', 'border': f'1px solid {SLATE2}',
                'padding': '16px 20px', 'flex': '1',
            }),
        ], style={'display': 'flex', 'gap': '12px', 'marginBottom': '12px'}),

        # Fila 2: verificación t+10
        html.Div([
            html.Div([
                html.Span('VERIFICACIÓN — 10 DÍAS HÁBILES POSTERIORES A LA SEÑAL', style={
                    'fontSize': '10.5px', 'color': SLATE5,
                    'textTransform': 'uppercase', 'letterSpacing': '0.07em', 'fontWeight': '500',
                }),
                info_icon(
                    'Muestra los 10 días hábiles después de la fecha seleccionada. '
                    'Permite comprobar si el precio efectivamente se movió en la dirección '
                    'que el modelo predijo. El porcentaje a la derecha es el retorno real.'
                ),
            ], style={'padding': '14px 20px 0'}),
            dcc.Loading(
                dcc.Graph(id='verification-chart', config={'displayModeBar': False},
                          style={'height': '240px'}),
                type='dot', color=ACCENT,
            ),
        ], style={
            'background': 'white', 'border': f'1px solid {SLATE2}', 'marginBottom': '12px',
        }),

        # Fila 3: indicadores
        html.Div([
            html.Div([
                html.Button('Técnico',       id='tab-tech', n_clicks=0, style=_tab_active),
                html.Button('Futuros & VIX', id='tab-fvix', n_clicks=0, style=_tab_inactive),
                html.Button('Fundamentales', id='tab-fund', n_clicks=0, style=_tab_inactive),
            ], style={
                'display': 'flex', 'gap': '4px', 'padding': '14px 20px 10px',
                'background': 'white', 'borderBottom': f'1px solid {SLATE2}',
            }),
            html.Div(id='tab-description'),
            dcc.Loading(html.Div([
                html.Div(id='tab-content-tech', style={
                    'display': 'grid', 'gridTemplateColumns': '1fr 1fr',
                    'gap': '1px', 'background': SLATE2,
                }),
                html.Div(id='tab-content-fvix', style={'display': 'none', 'background': 'white'}),
                html.Div(id='tab-content-fund', style={
                    'display': 'none', 'background': 'white', 'padding': '16px 20px',
                }),
            ]), type='dot', color=ACCENT),
        ], style={
            'background': 'white', 'border': f'1px solid {SLATE2}', 'marginBottom': '12px',
        }),

        # ── RESULTADOS GLOBALES ───────────────────────────────────────────────
        results_divider(),

        # Test 2023-2024
        html.Div([
            section_header(
                'PERÍODO DE TEST 2023–2024',
                f'{TEST_EQ["dates"][0]} → {TEST_EQ["dates"][-1]} · {len(TEST_EQ["dates"])} días de trading'
            ),
            html.Div([
                metric_card('Sharpe (Modelo 3-Class)',
                            f"{TEST_EQ['sharpe']:+.3f}",
                            f"B&H {TEST_EQ['sharpe_bh']:+.3f}  |  SPY {TEST_EQ['sharpe_spy']:+.3f}"),
                metric_card('Retorno Acumulado',
                            f"{TEST_EQ['retorno']*100:+.1f}%",
                            f"B&H {TEST_EQ['retorno_bh']*100:+.1f}%  |  SPY {TEST_EQ['retorno_spy']*100:+.1f}%"),
                metric_card('Máx. Caída (Drawdown)', f"{TEST_EQ['maxdd']*100:.1f}%"),
                metric_card('Tickers que superan B&H',
                            f"{TEST_EQ['n_beat']} / 10", 'Por Sharpe en período test'),
            ], style={'display': 'flex', 'gap': '1px', 'background': SLATE2, 'margin': '8px 0 1px'}),
            dcc.Graph(
                figure=fig_equity(TEST_EQ, title='Equity Curve Test 2023-2024 — Modelo vs B&H vs SPY (base 1 USD)'),
                config={'displayModeBar': False}, style={'height': '260px'},
            ),
        ], style={
            'background': 'white', 'border': f'1px solid {SLATE2}', 'marginBottom': '12px',
        }),

        # OOS 2025
        html.Div([
            section_header(
                'RENDIMIENTO OOS 2025',
                f'{OOS_EQ["dates"][0]} → {OOS_EQ["dates"][-1]} · {len(OOS_EQ["dates"])} días de trading'
            ),
            html.Div([
                metric_card('Sharpe — 3-Class',
                            f"{OOS_EQ['sharpe']:+.3f}",
                            f"B&H {OOS_EQ['sharpe_bh']:+.3f}  |  SPY {OOS_EQ['sharpe_spy']:+.3f}",
                            f"Binario {OOS_BIN['sharpe']:+.3f}"),
                metric_card('Retorno Acumulado',
                            f"{OOS_EQ['retorno']*100:+.1f}%",
                            f"B&H {OOS_EQ['retorno_bh']*100:+.1f}%  |  SPY {OOS_EQ['retorno_spy']*100:+.1f}%",
                            f"Binario {OOS_BIN['retorno']*100:+.1f}%"),
                metric_card('Máx. Caída — 3-Class',
                            f"{OOS_EQ['maxdd']*100:.1f}%",
                            f"Binario {OOS_BIN['maxdd']*100:.1f}%"),
                metric_card('Tickers que superan B&H',
                            f"{OOS_EQ['n_beat']} / 10",
                            'Por Sharpe — Modelo 3-Class',
                            f"Binario: {OOS_BIN['n_beat']} / 10"),
            ], style={'display': 'flex', 'gap': '1px', 'background': SLATE2, 'margin': '8px 0 1px'}),
            dcc.Graph(
                figure=fig_equity(OOS_EQ, OOS_BIN,
                                  title='Equity Curve OOS 2025 — Modelos vs B&H vs SPY (base 1 USD)'),
                config={'displayModeBar': False}, style={'height': '280px'},
            ),
        ], style={
            'background': 'white', 'border': f'1px solid {SLATE2}', 'marginBottom': '12px',
        }),

        html.Div(
            'Grupo 12 · MIAD Uniandes · Módulo 3 · Modelo XGBoost 3-Class ±1.5%',
            style={'textAlign': 'center', 'padding': '10px 0 16px',
                   'color': SLATE4, 'fontSize': '11px'},
        ),

    ], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '16px 20px'}),

], style={'minHeight': '100vh', 'background': SLATE1})

# ── callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output('date-picker', 'min_date_allowed'),
    Output('date-picker', 'max_date_allowed'),
    Output('date-picker', 'date'),
    Input('ticker-dropdown', 'value'),
)
def update_date_range(ticker):
    dates = df_test[df_test['Ticker'] == ticker]['Date'].sort_values()
    return (
        dates.iloc[0].strftime('%Y-%m-%d'),
        dates.iloc[-1].strftime('%Y-%m-%d'),
        dates.iloc[-11].strftime('%Y-%m-%d'),
    )


@app.callback(
    Output('store-result',  'data'),
    Output('header-status', 'children'),
    Input('ticker-dropdown', 'value'),
    Input('date-picker',     'date'),
)
def update_prediction(ticker, date):
    if not date or not ticker:
        raise PreventUpdate
    try:
        row, actual_date = _lookup_row(ticker, date)
        X      = pd.DataFrame([row[FEAT].to_dict()])
        result = predict(X, _cache, compute_shap=True)

        # filtrar NaN antes de serializar a JSON
        shap_clean = {
            k: v for k, v in (result['shap_values'] or {}).items()
            if v is not None and not (isinstance(v, float) and np.isnan(v))
        }
        shap_top = top_shap_features(shap_clean, n=10) if shap_clean else []

        fut_idx = ohlcv[ticker].index[ohlcv[ticker].index > actual_date][:10]
        t10     = fut_idx[-1].strftime('%d %b %Y') if len(fut_idx) == 10 else '?'

        return {
            'ticker'  : ticker,
            'signal'  : result['signal'],
            'p_sell'  : result['p_sell'],
            'p_hold'  : result['p_hold'],
            'p_buy'   : result['p_buy'],
            'date'    : actual_date.strftime('%Y-%m-%d'),
            'date_fmt': actual_date.strftime('%d %b %Y'),
            't10'     : t10,
            'shap_top': shap_top,
        }, f'{ticker} · {actual_date.strftime("%d %b %Y")}'
    except Exception as e:
        return None, f'Error: {str(e)[:50]}'


@app.callback(
    Output('signal-badge',   'children'),
    Output('signal-desc',    'children'),
    Output('signal-date',    'children'),
    Output('signal-horizon', 'children'),
    Output('prob-bars',      'children'),
    Output('shap-table',     'children'),
    Input('store-result', 'data'),
)
def render_signal(data):
    if data is None:
        return badge('—'), 'Selecciona un ticker y una fecha.', 'Fecha: —', '', [], html.Div()
    signal = data['signal']
    bars   = [
        prob_bar('BUY',  data['p_buy'],  BUY_C),
        prob_bar('HOLD', data['p_hold'], HOLD_C),
        prob_bar('SELL', data['p_sell'], SELL_C),
    ]
    return (
        badge(signal),
        SIGNAL_DESC.get(signal, ''),
        f"Fecha de entrada: {data['date_fmt']}",
        f"Horizonte: {data['date_fmt']} → {data['t10']} (≈10 días hábiles)",
        bars,
        shap_table(data['shap_top']),
    )


@app.callback(
    Output('verification-chart', 'figure'),
    Input('store-result', 'data'),
)
def render_verification(data):
    if data is None:
        return _fig_empty(240)
    return fig_verification(data['ticker'], data['date'], data['signal'])


@app.callback(
    Output('tab-content-tech', 'children'),
    Output('tab-content-fvix', 'children'),
    Output('tab-content-fund', 'children'),
    Input('store-result', 'data'),
)
def update_indicators(data):
    cell = {'background': 'white', 'padding': '4px'}
    if data is None:
        return [], html.Div(), html.Div()

    ticker = data['ticker']
    date   = data['date']

    try:
        df_ind = _prep_indicators_df(ticker, date)

        tech = [
            html.Div(dcc.Graph(figure=fig_candlestick_context(df_ind, date),
                               config={'displayModeBar': False},
                               style={'height': '220px'}), style=cell),
            html.Div(dcc.Graph(figure=fig_rsi(df_ind, date),
                               config={'displayModeBar': False},
                               style={'height': '220px'}), style=cell),
            html.Div(dcc.Graph(figure=fig_macd(df_ind, date),
                               config={'displayModeBar': False},
                               style={'height': '220px'}), style=cell),
            html.Div(dcc.Graph(figure=fig_bollinger(df_ind, date),
                               config={'displayModeBar': False},
                               style={'height': '220px'}), style=cell),
        ]

        fvix = html.Div([
            dcc.Graph(figure=fig_vix_context(date, df_ind),
                      config={'displayModeBar': False},
                      style={'height': '220px', 'borderBottom': f'1px solid {SLATE2}'}),
            dcc.Graph(figure=fig_futures_context(date, df_ind),
                      config={'displayModeBar': False},
                      style={'height': '220px'}),
        ], style={'background': 'white'})

        row, _ = _lookup_row(ticker, date)
        fund_rows = []
        for col, lbl, fmt in FUND_FIELDS:
            if col not in row.index:
                continue
            val = row[col]
            try:
                display = fmt.format(val) if pd.notna(val) else '—'
            except Exception:
                display = '—'
            fund_rows.append((lbl, display))
        fund = fund_table(fund_rows)

        return tech, fvix, fund

    except Exception as e:
        msg = html.Div(f'Error: {e}',
                       style={'color': SELL_C, 'padding': '20px', 'fontSize': '12px'})
        return [msg], msg, msg


@app.callback(
    Output('tab-content-tech', 'style'),
    Output('tab-content-fvix', 'style'),
    Output('tab-content-fund', 'style'),
    Output('tab-tech',         'style'),
    Output('tab-fvix',         'style'),
    Output('tab-fund',         'style'),
    Output('tab-description',  'children'),
    Input('tab-tech', 'n_clicks'),
    Input('tab-fvix', 'n_clicks'),
    Input('tab-fund', 'n_clicks'),
)
def switch_tab(n_tech, n_fvix, n_fund):
    grid   = {'display': 'grid', 'gridTemplateColumns': '1fr 1fr',
               'gap': '1px', 'background': SLATE2}
    col    = {'display': 'block', 'background': 'white'}
    block  = {'display': 'block', 'background': 'white', 'padding': '16px 20px'}
    hidden = {'display': 'none'}

    n_tech = n_tech or 0
    n_fvix = n_fvix or 0
    n_fund = n_fund or 0

    if n_fund > n_tech and n_fund > n_fvix:
        return hidden, hidden, block, _tab_inactive, _tab_inactive, _tab_active, tab_description(
            'Datos financieros de la empresa: rentabilidad, valoración y estructura de capital. '
            'Son de contexto — el modelo no los usa directamente como variables de entrada.'
        )
    if n_fvix > n_tech:
        return hidden, col, hidden, _tab_inactive, _tab_active, _tab_inactive, tab_description(
            'El VIX mide el nerviosismo del mercado: por encima de 20 indica alta incertidumbre. '
            'ES=F es el contrato de futuros del S&P 500 y anticipa la dirección del mercado.'
        )
    return grid, hidden, hidden, _tab_active, _tab_inactive, _tab_inactive, tab_description(
        f'Indicadores calculados a partir del precio y volumen histórico ({N_DISPLAY} días). '
        'El RSI mide si la acción está sobrecomprada (>70) o sobrevendida (<30). '
        'El MACD detecta cambios de tendencia. Las Bandas de Bollinger muestran la volatilidad.'
    )


server = app.server  # exponer Flask para gunicorn

if __name__ == '__main__':
    port  = int(os.environ.get('PORT', 8050))
    debug = os.environ.get('RENDER') is None
    app.run(debug=debug, host='0.0.0.0', port=port)
