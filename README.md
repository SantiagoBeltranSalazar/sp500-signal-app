# S&P 500 · Signal Model — Prototipo

Dashboard de recomendaciones de inversión basado en XGBoost 3-Class para 10 acciones del S&P 500.
Proyecto de Grado · MIAD Uniandes · Grupo 20.

## Requisitos

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) para gestión de dependencias

## Instalación

```bash
uv sync
```

## Correr el dashboard

```bash
uv run python src/dashboard/app.py
```

Abre `http://localhost:8050` en el navegador.

## Correr la API (opcional)

```bash
uv run uvicorn src.api.main:app --reload
```

Documentación interactiva en `http://localhost:8000/docs`.

## Tickers soportados

AAPL · MSFT · NVDA · GOOGL · AMZN · META · TSLA · JPM · UNH · XOM

## Modelo

`models/final/xgboost_3class_main.pkl` — XGBoost multiclase calibrado con regresión isotónica.
Señales: **BUY** (+1) · **HOLD** (0) · **SELL** (−1). Umbral ±1.5% sobre retorno acumulado a 10 días.
