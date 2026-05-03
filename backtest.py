"""
TradingAgents Backtest Script
Corre propagate() en múltiples fechas y calcula retornos reales vs SPY.
"""

import os
import json
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

import yfinance as yf
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# ─────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────
TICKER        = "AAPL"          # Ticker a analizar
START_DATE    = "2024-07-01"    # Fecha inicio del backtest
END_DATE      = "2024-12-01"    # Fecha fin del backtest
FREQ_DAYS     = 30              # Cada cuántos días analizar
HOLDING_DAYS  = 5               # Días que se mantiene la posición
ANALYSTS      = ["market", "fundamentals"]  # Solo 2 para reducir costo
# ─────────────────────────────────────────

def generate_dates(start: str, end: str, freq: int):
    dates = []
    current = datetime.strptime(start, "%Y-%m-%d")
    end_dt  = datetime.strptime(end,   "%Y-%m-%d")
    while current <= end_dt:
        # Saltar fines de semana
        if current.weekday() < 5:
            dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=freq)
    return dates


def get_return(ticker: str, trade_date: str, holding_days: int):
    """Retorno real de la acción y alpha vs SPY."""
    try:
        start = datetime.strptime(trade_date, "%Y-%m-%d")
        end   = start + timedelta(days=holding_days + 7)
        end_s = end.strftime("%Y-%m-%d")

        stock = yf.Ticker(ticker).history(start=trade_date, end=end_s)
        spy   = yf.Ticker("SPY").history(start=trade_date, end=end_s)

        if len(stock) < 2 or len(spy) < 2:
            return None, None

        n     = min(holding_days, len(stock) - 1, len(spy) - 1)
        raw   = (stock["Close"].iloc[n] - stock["Close"].iloc[0]) / stock["Close"].iloc[0]
        bench = (spy["Close"].iloc[n]   - spy["Close"].iloc[0])   / spy["Close"].iloc[0]
        return float(raw), float(raw - bench)
    except Exception as e:
        print(f"  [!] Error calculando retorno: {e}")
        return None, None


def signal_to_position(decision: str) -> int:
    """BUY=1, HOLD=0, SELL=-1"""
    d = decision.lower()
    if any(x in d for x in ["buy", "overweight", "strong buy"]):
        return 1
    if any(x in d for x in ["sell", "underweight", "strong sell"]):
        return -1
    return 0


def main():
    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"]      = 1
    config["max_risk_discuss_rounds"] = 1

    ta = TradingAgentsGraph(
        selected_analysts=ANALYSTS,
        config=config,
        debug=False
    )

    dates   = generate_dates(START_DATE, END_DATE, FREQ_DAYS)
    results = []

    print(f"\n{'='*60}")
    print(f"  BACKTEST: {TICKER} | {START_DATE} → {END_DATE}")
    print(f"  Fechas: {len(dates)} | Holding: {HOLDING_DAYS}d | Analistas: {ANALYSTS}")
    print(f"{'='*60}\n")

    for i, date in enumerate(dates, 1):
        print(f"[{i}/{len(dates)}] Analizando {TICKER} @ {date}...", end=" ", flush=True)
        try:
            _, decision = ta.propagate(TICKER, date)
            position    = signal_to_position(decision)
            raw, alpha  = get_return(TICKER, date, HOLDING_DAYS)

            pnl = (raw * position) if raw is not None else None

            result = {
                "date":     date,
                "decision": decision,
                "position": position,
                "raw_ret":  round(raw   * 100, 2) if raw   is not None else None,
                "alpha":    round(alpha * 100, 2) if alpha is not None else None,
                "pnl":      round(pnl   * 100, 2) if pnl   is not None else None,
            }
            results.append(result)

            pos_str = "BUY 📈" if position == 1 else ("SELL 📉" if position == -1 else "HOLD ➡️")
            ret_str = f"ret={result['raw_ret']:+.1f}%  alpha={result['alpha']:+.1f}%  pnl={result['pnl']:+.1f}%" if raw else "retorno pendiente"
            print(f"{decision:<12} → {pos_str}  |  {ret_str}")

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"date": date, "decision": "ERROR", "position": 0,
                            "raw_ret": None, "alpha": None, "pnl": None})

        if i < len(dates):
            time.sleep(2)  # Pausa entre llamadas

    # ─── Resumen ───────────────────────────────────────────
    valid = [r for r in results if r["pnl"] is not None]
    if valid:
        total_pnl    = sum(r["pnl"]    for r in valid)
        total_alpha  = sum(r["alpha"]  for r in valid)
        wins         = sum(1 for r in valid if r["pnl"] > 0)
        win_rate     = wins / len(valid) * 100
        buys         = sum(1 for r in valid if r["position"] ==  1)
        holds        = sum(1 for r in valid if r["position"] ==  0)
        sells        = sum(1 for r in valid if r["position"] == -1)

        print(f"\n{'='*60}")
        print(f"  RESUMEN BACKTEST — {TICKER}")
        print(f"{'='*60}")
        print(f"  Períodos analizados : {len(valid)}")
        print(f"  BUY / HOLD / SELL   : {buys} / {holds} / {sells}")
        print(f"  Win rate            : {win_rate:.1f}%")
        print(f"  PnL total acumulado : {total_pnl:+.2f}%")
        print(f"  Alpha total vs SPY  : {total_alpha:+.2f}%")
        print(f"  PnL promedio/trade  : {total_pnl/len(valid):+.2f}%")
        print(f"{'='*60}\n")

    # Guardar resultados en JSON
    out_file = f"backtest_{TICKER}_{START_DATE}_{END_DATE}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"ticker": TICKER, "config": {"analysts": ANALYSTS,
                   "holding_days": HOLDING_DAYS}, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"Resultados guardados en: {out_file}")


if __name__ == "__main__":
    main()
