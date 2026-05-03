# TradingGrowth — Contexto del Proyecto

## Qué es esto
Fork personalizado de [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents).
Framework multi-agente LLM para análisis financiero. Repo: https://github.com/SantiagoBoveda/TradingGrowth

## Stack
- **Python 3.12** — `C:\Users\santi\AppData\Local\Programs\Python\Python312\python.exe`
- **LLM:** Google Gemini 2.5 Flash (`google` provider)
- **Datos:** yfinance (fundamentals) + Alpha Vantage (precios/noticias)
- **Framework:** LangGraph + LangChain
- **Ubicación:** `C:\Users\santi\Dropbox\Claude\TradingGrowth\TradingAgents\`

## Configuración actual (`tradingagents/default_config.py`)
```python
"llm_provider": "google"
"deep_think_llm": "gemini-2.5-flash"
"quick_think_llm": "gemini-2.5-flash"
"output_language": "Spanish"
"max_debate_rounds": 2
"max_risk_discuss_rounds": 2
"checkpoint_enabled": True
"data_vendors": {
    "core_stock_apis": "yfinance",
    "technical_indicators": "yfinance",
    "fundamental_data": "yfinance",
    "news_data": "yfinance",
}
```
> Alpha Vantage free tier solo funciona para precios básicos — fundamentals son premium. Por eso todo en yfinance.

## API Keys (en `.env`, nunca commitear)
- `GOOGLE_API_KEY` — Gemini (proyecto con billing habilitado, spend cap subido)
- `ALPHA_VANTAGE_API_KEY` — plan gratuito (25 req/día)

## Cómo correr un análisis
```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
ta = TradingAgentsGraph(
    selected_analysts=['market', 'social', 'news', 'fundamentals'],
    config=config,
    debug=False
)
_, decision = ta.propagate('AAPL', '2025-01-15')
print(decision)
```

## Costo por análisis
- ~$1 USD con 4 agentes + 2 rondas de debate (Gemini 2.5 Flash con thinking)
- ~$0.10-0.20 con 2 agentes (market + fundamentals) + 1 ronda

## Modificaciones realizadas

### 1. `tradingagents/agents/analysts/fundamentals_analyst.py`
- Agregado `get_insider_transactions` a las tools
- Prompt reescrito con 10 secciones obligatorias:
  1. Valuation Analysis (P/E, PEG, P/B, EV/EBITDA, P/FCF)
  2. Profitability & Margins (gross/op/net margin, ROE, ROA, ROIC)
  3. Revenue & Earnings Quality (YoY/QoQ, accruals, Beneish M-Score)
  4. Balance Sheet Strength (D/E, Net Debt, Current Ratio, Interest Coverage)
  5. Cash Flow Analysis (FCF, CapEx intensity, FCF yield, dividend sustainability)
  6. Capital Allocation (dividends, buybacks, acquisitions)
  7. Insider Activity (compras/ventas recientes)
  8. Red Flags & Risks
  9. Bull vs Bear Case
  10. Final Fundamental Verdict con target price range

### 2. `tradingagents/default_config.py`
- Provider cambiado a Google Gemini
- Output en español
- Checkpoint habilitado
- 2 rondas de debate

### 3. `backtest.py` (nuevo archivo)
- Backtest multi-fecha con `propagate()` en loop
- Calcula raw return y alpha vs SPY por cada decisión
- BUY=long, SELL=short, HOLD=fuera
- Muestra win rate, PnL acumulado, alpha total
- Guarda resultados en JSON
- Config editable en las primeras líneas (TICKER, START_DATE, END_DATE, FREQ_DAYS, HOLDING_DAYS)

## Análisis realizados
| Ticker | Fecha | Agentes | Decisión | Razón principal |
|--------|-------|---------|----------|----------------|
| AAPL | 2025-01-15 | market + news | HOLD | Sin catalizador, momentum bajista leve |
| AAPL | 2025-01-15 | 4 agentes | UNDERWEIGHT | Aprendió de HOLD anterior (-8.8% alpha) |
| CVX | 2025-01-15 | 4 agentes | SELL | Deuda triplicada, pagos > FCF |
| GOLD | 2025-01-15 | 4 agentes | SELL | P/E 88x, beneficio -56%, deuda = 88x FCF |

## Próximos pasos sugeridos
1. **Correr backtest** — `python backtest.py` (configurar ticker/fechas al inicio del archivo)
2. **Mejorar market analyst** — agregar Fibonacci, OBV, Volume Profile al prompt
3. **Mejorar news analyst** — integrar NewsAPI o Finnhub para noticias reales
4. **Interfaz web** — dashboard para ingresar ticker y ver análisis
5. **Automatización** — análisis diario de una lista de tickers con alertas

## Problemas conocidos
- Alpha Vantage free tier: solo `core_stock_apis` funciona, el resto es premium
- Gemini spend cap: si falla con 429 RESOURCE_EXHAUSTED, subir cap en ai.studio/spend
- OpenRouter modelos `:free` — frecuentemente saturados, no confiable para producción
- Social analyst: datos limitados con yfinance (pocas noticias de redes sociales)

## Comandos útiles
```bash
# Correr análisis
cd C:\Users\santi\Dropbox\Claude\TradingGrowth\TradingAgents
C:\Users\santi\AppData\Local\Programs\Python\Python312\python.exe -c "..."

# Correr backtest
C:\Users\santi\AppData\Local\Programs\Python\Python312\python.exe backtest.py

# CLI interactiva
C:\Users\santi\AppData\Local\Programs\Python\Python312\python.exe -m cli.main

# Push a GitHub
git add -A && git commit -m "mensaje" && git push
```
