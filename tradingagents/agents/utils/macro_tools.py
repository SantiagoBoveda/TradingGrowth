"""
Macro analysis tools — VIX, rates, DXY, sector rotation.
No API key required (yfinance only).
"""
from langchain_core.tools import tool
from typing import Annotated


@tool
def get_macro_indicators(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
) -> str:
    """
    Fetch key macro indicators: VIX fear index, 10yr Treasury yield,
    US Dollar Index (DXY), S&P 500, NASDAQ, Gold, and Bond ETF.
    Returns 1-day and 1-month price changes for each.
    """
    import yfinance as yf
    from datetime import datetime, timedelta
    try:
        end = datetime.strptime(curr_date, "%Y-%m-%d")
        start = (end - timedelta(days=35)).strftime("%Y-%m-%d")

        tickers = {
            "^VIX":     "VIX (Fear Index)",
            "^TNX":     "10yr Treasury Yield",
            "DX-Y.NYB": "US Dollar Index (DXY)",
            "^GSPC":    "S&P 500",
            "^IXIC":    "NASDAQ",
            "GLD":      "Gold ETF",
            "TLT":      "20yr Treasury Bond ETF",
            "^RUT":     "Russell 2000 (Small Caps)",
        }

        lines = [f"=== Macro Indicators as of {curr_date} ===\n"]
        for symbol, name in tickers.items():
            try:
                h = yf.Ticker(symbol).history(start=start, end=curr_date)
                if len(h) < 2:
                    continue
                latest   = h["Close"].iloc[-1]
                prev_day = h["Close"].iloc[-2]
                prev_mo  = h["Close"].iloc[0]
                d_chg = (latest - prev_day) / prev_day * 100
                m_chg = (latest - prev_mo)  / prev_mo  * 100
                lines.append(
                    f"{name} ({symbol}): {latest:.2f}  |  1d: {d_chg:+.2f}%  |  1mo: {m_chg:+.2f}%"
                )
            except Exception:
                continue

        # VIX regime interpretation
        try:
            vix_val = yf.Ticker("^VIX").history(start=start, end=curr_date)["Close"].iloc[-1]
            if vix_val < 15:
                regime = "LOW FEAR — complacent market, potential risk-on"
            elif vix_val < 25:
                regime = "MODERATE — normal market conditions"
            elif vix_val < 35:
                regime = "ELEVATED FEAR — cautious, increased volatility"
            else:
                regime = "EXTREME FEAR — crisis conditions, potential capitulation"
            lines.append(f"\nVIX Regime: {regime}")
        except Exception:
            pass

        return "\n".join(lines)
    except Exception as e:
        return f"Macro indicators error: {e}"


@tool
def get_sector_performance(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    lookback_days: Annotated[int, "Days to look back for performance"] = 30,
) -> str:
    """
    Fetch 11 SPDR sector ETF performance to identify sector rotation trends.
    Returns sectors ranked by 1-month performance.
    """
    import yfinance as yf
    from datetime import datetime, timedelta
    try:
        end  = datetime.strptime(curr_date, "%Y-%m-%d")
        start = (end - timedelta(days=lookback_days + 5)).strftime("%Y-%m-%d")

        sectors = {
            "XLK":  "Technology",
            "XLF":  "Financials",
            "XLE":  "Energy",
            "XLV":  "Healthcare",
            "XLY":  "Consumer Discretionary",
            "XLP":  "Consumer Staples",
            "XLI":  "Industrials",
            "XLB":  "Materials",
            "XLU":  "Utilities",
            "XLRE": "Real Estate",
            "XLC":  "Communication Services",
        }

        perf = []
        for etf, name in sectors.items():
            try:
                h = yf.Ticker(etf).history(start=start, end=curr_date)
                if len(h) < 2:
                    continue
                chg = (h["Close"].iloc[-1] - h["Close"].iloc[0]) / h["Close"].iloc[0] * 100
                perf.append((name, etf, chg))
            except Exception:
                continue

        perf.sort(key=lambda x: x[2], reverse=True)

        lines = [f"=== Sector Performance — last {lookback_days} days (as of {curr_date}) ===\n"]
        lines.append("LEADERS (outperforming):")
        for name, etf, chg in perf[:3]:
            lines.append(f"  ↑ {name} ({etf}): {chg:+.2f}%")
        lines.append("\nMID:")
        for name, etf, chg in perf[3:-3]:
            lines.append(f"  → {name} ({etf}): {chg:+.2f}%")
        lines.append("\nLAGGARDS (underperforming):")
        for name, etf, chg in perf[-3:]:
            lines.append(f"  ↓ {name} ({etf}): {chg:+.2f}%")

        return "\n".join(lines)
    except Exception as e:
        return f"Sector performance error: {e}"
