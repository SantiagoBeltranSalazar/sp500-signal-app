"""
Script de preparación de datos — ejecutar una vez antes de lanzar el dashboard.

Qué hace:
  1. Descarga OHLCV 2022-2025 de los 10 tickers + SPY desde Yahoo Finance.
  2. Copia test.parquet y master_features_oos.parquet desde la ruta local de Proyecto 1.

Uso:
  uv run python scripts/prepare_data.py
  uv run python scripts/prepare_data.py --features-src /ruta/alternativa/a/master/
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd
import yfinance as yf

TICKERS = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'JPM', 'UNH', 'XOM']
BENCH   = ['SPY']
START   = '2022-01-01'
END     = '2025-12-31'

REPO_ROOT    = Path(__file__).parent.parent
OHLCV_DIR    = REPO_ROOT / 'data' / 'ohlcv'
FEATURES_DIR = REPO_ROOT / 'data' / 'features'

# ruta por defecto en el equipo de Santiago
DEFAULT_FEATURES_SRC = Path(
    '/Users/santiagobeltran/Desktop/'
    'Master en Inteligencia Analítica de Datos/'
    'Cuarto semestre/Segundo ciclo/Proyecto de Grado/'
    'Proyecto 1/data/master'
)


def descargar_ohlcv(tickers: list[str]) -> None:
    OHLCV_DIR.mkdir(parents=True, exist_ok=True)
    for ticker in tickers:
        print(f'  descargando {ticker}...')
        raw = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
        if raw.empty:
            print(f'  ADVERTENCIA: sin datos para {ticker}')
            continue
        # aplanar columnas multi-nivel si vienen así
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df.index.name = 'Date'
        df['Ticker'] = ticker
        dest = OHLCV_DIR / f'{ticker}_ohlcv.parquet'
        df.to_parquet(dest)
        print(f'  {ticker}: {df.index.min().date()} → {df.index.max().date()} ({len(df)} filas) → {dest.name}')


def copiar_features(src: Path) -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    archivos = {
        'test.parquet':                 'test.parquet',
        'master_features_oos.parquet':  'master_features_oos.parquet',
    }
    for origen, destino in archivos.items():
        ruta_origen = src / origen
        ruta_destino = FEATURES_DIR / destino
        if not ruta_origen.exists():
            print(f'  ADVERTENCIA: no encontrado → {ruta_origen}')
            continue
        shutil.copy2(ruta_origen, ruta_destino)
        df = pd.read_parquet(ruta_destino)
        print(f'  {destino}: {len(df)} filas copiadas → {ruta_destino}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--features-src',
        type=Path,
        default=DEFAULT_FEATURES_SRC,
        help='Carpeta con test.parquet y master_features_oos.parquet'
    )
    args = parser.parse_args()

    print('\n=== Paso 1: OHLCV (10 tickers + SPY) ===')
    descargar_ohlcv(TICKERS + BENCH)

    print(f'\n=== Paso 2: features desde {args.features_src} ===')
    copiar_features(args.features_src)

    print('\nListo. Datos en data/ohlcv/ y data/features/')


if __name__ == '__main__':
    main()
