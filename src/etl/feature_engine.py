# feature_engine.py — calcula las 30 features que espera xgboost_3class_main.pkl
# Modularización de notebooks/02_oos_etl_2025.ipynb

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator

warnings.filterwarnings('ignore')

TICKERS = ['AAPL','MSFT','NVDA','GOOGL','AMZN','META','TSLA','JPM','UNH','XOM']

SHEET_MAP = {'APPL':'AAPL','MSFT':'MSFT','GOOGL':'GOOGL','AMZN':'AMZN','META':'META',
             'TSLA':'TSLA','JPM':'JPM','UNH':'UNH','XOM':'XOM'}

FEAT_MAP = {
    'HIST_CALL_IMP_VOL'             : 'opt_impl_vol_call',
    'HIST_PUT_IMP_VOL'              : 'opt_impl_vol_put',
    'VOLATILITY_10D'                : 'opt_hist_vol_10d',
    'VOLATILITY_30D'                : 'opt_hist_vol_30d',
    'PUT_CALL_VOLUME_RATIO_CUR_DAY' : 'opt_put_call_vol_ratio',
    'PUT_CALL_OPEN_INTEREST_RATIO'  : 'opt_put_call_oi_ratio',
    'IVOL_DELTA'                    : 'opt_ivol_delta',
}

# features precio-dependientes → normalizar por Close → sufijo _rel
PRICE_REL = {
    'SMA_20':'SMA_20_rel', 'SMA_50':'SMA_50_rel', 'EMA_12':'EMA_12_rel', 'EMA_26':'EMA_26_rel',
    'BB_upper':'BB_upper_rel', 'BB_middle':'BB_middle_rel', 'BB_lower':'BB_lower_rel',
    'MACD':'MACD_rel', 'MACD_signal':'MACD_signal_rel', 'MACD_diff':'MACD_diff_rel',
    'ATR_14':'ATR_14_rel',
}

# las 30 features en el orden exacto que espera el pkl
FEATURE_COLS = [
    'OBV', 'Volume_change', 'log_return_lag2', 'log_return_lag1', 'opt_ivol_vs_vix',
    'Volume', 'VIX_MACD_Signal', 'BB_lower_rel', 'opt_ivol_delta', 'RSI_14',
    'Volatility_10', 'EMA_12_rel', 'opt_ivol_vs_hvol', 'BB_width', 'MACD_diff_rel',
    'Spread_Future_Spot', 'log_return_lag5', 'Future_RSI_14', 'Volume_Change',
    'opt_hist_vol_30d', 'Future_Returns', 'opt_put_call_vol_ratio', 'Future_Close',
    'MACD_signal_rel', 'opt_impl_vol_call', 'opt_put_call_oi_ratio', 'Future_SMA_20',
    'Future_Volatility_10', 'VIX_SMA_5', 'ATR_14_rel',
]


def rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0).rolling(p).mean()
    l = (-d.where(d < 0, 0)).rolling(p).mean()
    return 100 - (100 / (1 + g/l))


def macd_calc(s, fast=12, slow=26, sig=9):
    ef = s.ewm(span=fast, adjust=False).mean()
    es = s.ewm(span=slow, adjust=False).mean()
    m  = ef - es
    sg = m.ewm(span=sig, adjust=False).mean()
    return m, sg, m - sg


def calc_tech_indicators(df):
    df = df.copy()
    df['SMA_20'] = SMAIndicator(df['Close'], 20).sma_indicator()
    df['SMA_50'] = SMAIndicator(df['Close'], 50).sma_indicator()
    df['EMA_12'] = EMAIndicator(df['Close'], 12).ema_indicator()
    df['EMA_26'] = EMAIndicator(df['Close'], 26).ema_indicator()
    df['RSI_14'] = RSIIndicator(df['Close'], 14).rsi()
    macd_ind      = MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD']        = macd_ind.macd()
    df['MACD_signal'] = macd_ind.macd_signal()
    df['MACD_diff']   = macd_ind.macd_diff()
    bb = BollingerBands(df['Close'], 20, 2)
    df['BB_upper']  = bb.bollinger_hband()
    df['BB_middle'] = bb.bollinger_mavg()
    df['BB_lower']  = bb.bollinger_lband()
    df['BB_width']  = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    df['ATR_14']    = AverageTrueRange(df['High'], df['Low'], df['Close'], 14).average_true_range()
    df['OBV']       = OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    df['Returns']       = df['Close'].pct_change()
    df['Volatility_10'] = df['Returns'].rolling(10).std()
    df['Volume_change'] = df['Volume'].pct_change()
    return df


def download_tech(tickers, start, end):
    blocks = []
    for tk in tickers:
        df = yf.download(tk, start=start, end=end, progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df = calc_tech_indicators(df)
        df.insert(0, 'Ticker', tk)
        # eliminar OHLCV bruto salvo Close, que se necesita para normalizar _rel después
        drop = [c for c in ['Open','High','Low','Volume','Dividends','Stock Splits','Adj Close'] if c in df.columns]
        df = df.drop(columns=drop)
        blocks.append(df)
        print(f'{tk:6s} | {len(df):4d} días | {df["Date"].min().date()} → {df["Date"].max().date()}')
    return pd.concat(blocks, ignore_index=True)


def download_vix(start, end):
    vix_raw = yf.download('^VIX', start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix_raw.columns = vix_raw.columns.get_level_values(0)
    vix_raw = vix_raw.reset_index()
    vix_raw['Date'] = pd.to_datetime(vix_raw['Date']).dt.tz_localize(None)

    vix = pd.DataFrame({'Date': vix_raw['Date'], 'VIX': vix_raw['Close']})
    vix['VIX_Change']        = vix['VIX'].diff()
    vix['VIX_Returns']       = vix['VIX'].pct_change()
    vix['VIX_SMA_5']         = vix['VIX'].rolling(5).mean()
    vix['VIX_SMA_20']        = vix['VIX'].rolling(20).mean()
    vix['VIX_Trend']         = vix['VIX'] - vix['VIX_SMA_20']
    vix['VIX_Volatility_10'] = vix['VIX_Returns'].rolling(10).std()
    vix['VIX_Regime']        = vix['VIX'].apply(lambda v: 0 if v < 15 else (1 if v < 25 else 2))
    vix['VIX_Spike']         = ((vix['VIX_Returns'] > 0.20) | (vix['VIX_Change'] > 5)).astype(int)
    vix['VIX_RSI_14']        = rsi(vix['VIX'], 14)
    vix['VIX_MACD'], vix['VIX_MACD_Signal'], vix['VIX_MACD_Diff'] = macd_calc(vix['VIX'])
    print(f'VIX shape: {vix.shape} | {vix["Date"].min().date()} → {vix["Date"].max().date()}')
    return vix


def download_futures(start, end):
    es  = yf.download('ES=F', start=start, end=end, progress=False, auto_adjust=False)
    spy_raw = yf.download('SPY', start=start, end=end, progress=False, auto_adjust=False)
    for d in (es, spy_raw):
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
    es      = es.reset_index();      es['Date']      = pd.to_datetime(es['Date']).dt.tz_localize(None)
    spy_raw = spy_raw.reset_index(); spy_raw['Date'] = pd.to_datetime(spy_raw['Date']).dt.tz_localize(None)
    spy = spy_raw[['Date','Close']].rename(columns={'Close':'SPY_Close'})

    fut = pd.DataFrame({'Date': es['Date']})
    fut['Future_Close']         = es['Close']
    fut['Future_Returns']       = fut['Future_Close'].pct_change()
    fut['Future_Volatility_10'] = fut['Future_Returns'].rolling(10).std()
    fut['Volume']               = es['Volume']
    fut['Volume_Change']        = es['Volume'].pct_change()
    fut = fut.merge(spy, on='Date', how='left')
    fut['SPY_Adjusted']       = fut['SPY_Close'] * 10
    fut['Spread_Future_Spot'] = fut['Future_Close'] - fut['SPY_Adjusted']
    fut['Spread_Pct']         = (fut['Spread_Future_Spot'] / fut['SPY_Adjusted']) * 100
    fut['Future_SMA_20']      = fut['Future_Close'].rolling(20).mean()
    fut['Future_RSI_14']      = rsi(fut['Future_Close'], 14)
    fut['Future_MACD'], fut['Future_MACD_Signal'], fut['Future_MACD_Diff'] = macd_calc(fut['Future_Close'])

    feat_cols = ['Future_Close','Future_Returns','Future_Volatility_10','Spread_Future_Spot','Spread_Pct',
                 'Volume','Volume_Change','Future_SMA_20','Future_RSI_14',
                 'Future_MACD','Future_MACD_Signal','Future_MACD_Diff']
    fut = fut[['Date'] + feat_cols]
    print(f'Futuros shape: {fut.shape} | {fut["Date"].min().date()} → {fut["Date"].max().date()}')
    return fut


def winsorize(df, cols, lower=0.01, upper=0.99):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            lo, hi = df[c].quantile(lower), df[c].quantile(upper)
            df[c] = df[c].clip(lo, hi)
    return df


def extract_sheet(filepath, sheet, ticker, start, end):
    raw = pd.read_excel(filepath, sheet_name=sheet, header=None)
    headers = raw.iloc[6].tolist()
    df = raw.iloc[7:].copy()
    df.columns = headers
    df = df.rename(columns={headers[0]: 'Date'})
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    df = df[(df['Date'] >= start) & (df['Date'] <= end)]
    df.insert(0, 'Ticker', ticker)
    return df


def load_bloomberg(filepath, start, end):
    all_blocks = []
    for sheet, ticker in SHEET_MAP.items():
        try:
            df_raw = extract_sheet(filepath, sheet, ticker, start, end)
        except Exception:
            continue
        cols_present = [c for c in FEAT_MAP if c in df_raw.columns]
        df_sel = df_raw[['Ticker','Date'] + cols_present].copy()
        rename = {k: v for k, v in FEAT_MAP.items() if k in df_raw.columns}
        df_sel = df_sel.rename(columns=rename)
        feat_cols = list(rename.values())
        df_sel[feat_cols] = df_sel[feat_cols].apply(pd.to_numeric, errors='coerce')
        df_sel = winsorize(df_sel, feat_cols)
        all_blocks.append(df_sel)
        print(f'{ticker:6s} | {len(df_sel):4d} días | NaN: {df_sel[feat_cols].isna().sum().sum()}')

    deriv = pd.concat(all_blocks, ignore_index=True)
    deriv['opt_impl_vol_spread'] = deriv['opt_impl_vol_call'] - deriv['opt_impl_vol_put']
    deriv['opt_ivol_vs_hvol']    = (deriv['opt_ivol_delta'] / deriv['opt_hist_vol_10d'].replace(0, np.nan)).replace([np.inf,-np.inf], np.nan)
    print(f'\nDerivados consolidado: {deriv.shape} | tickers: {sorted(deriv["Ticker"].unique())}')
    return deriv


def build_features(tickers=None, start='2024-01-01', end=None, output_start=None, bloomberg_file=None):
    if tickers is None:
        tickers = TICKERS
    if end is None:
        end = pd.Timestamp.today().strftime('%Y-%m-%d')

    print(f'Descargando OHLCV de los {len(tickers)} tickers...\n')
    tech = download_tech(tickers, start, end)

    print('\nDescargando VIX...')
    vix = download_vix(start, end)

    print('\nDescargando ES=F y SPY...')
    fut = download_futures(start, end)

    # normalizar features precio-dependientes (dividir por Close → _rel)
    base = tech.copy()
    for orig, rel in PRICE_REL.items():
        base[rel] = (base[orig] / base['Close'].replace(0, np.nan)).replace([np.inf,-np.inf], np.nan)
    base = base.drop(columns=list(PRICE_REL.keys()) + ['Close'])

    master = base.merge(vix, on='Date', how='left')
    master = master.merge(fut, on='Date', how='left')

    if bloomberg_file is not None:
        print('\nExtrayendo hojas Bloomberg...\n')
        deriv = load_bloomberg(bloomberg_file, start, end)
        master = master.merge(deriv, on=['Ticker','Date'], how='left')
    else:
        # sin Bloomberg → todas las opt_* quedan NaN; XGBoost lo maneja nativamente
        for col in ['opt_impl_vol_call','opt_impl_vol_put','opt_hist_vol_10d','opt_hist_vol_30d',
                    'opt_put_call_vol_ratio','opt_put_call_oi_ratio','opt_ivol_delta',
                    'opt_ivol_vs_hvol','opt_impl_vol_spread']:
            master[col] = np.nan

    master = master.sort_values(['Ticker','Date']).reset_index(drop=True)
    master['log_return'] = np.log1p(master['Returns'])
    for lag in [1,2,5]:
        master[f'log_return_lag{lag}'] = master.groupby('Ticker')['log_return'].shift(lag)

    master['opt_ivol_vs_vix'] = (master['opt_ivol_delta'] / master['VIX'].replace(0, np.nan)).replace([np.inf,-np.inf], np.nan)

    if output_start is not None:
        master = master[master['Date'] >= output_start].copy().reset_index(drop=True)

    missing = [f for f in FEATURE_COLS if f not in master.columns]
    assert not missing, f'Features faltantes en el master: {missing}'

    print(f'\nMaster final: {master.shape} | {master["Date"].min().date()} → {master["Date"].max().date()}')
    return master


def get_model_inputs(master):
    missing = [f for f in FEATURE_COLS if f not in master.columns]
    assert not missing, f'Columnas faltantes: {missing}'
    return master[['Ticker','Date'] + FEATURE_COLS].copy()
