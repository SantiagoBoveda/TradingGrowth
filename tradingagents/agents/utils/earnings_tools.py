"""
Earnings analysis tools — history, surprises, calendar, guidance.
Uses yfinance only (no API key required).
"""
from langchain_core.tools import tool
from typing import Annotated


@tool
def get_earnings_data(
    ticker: Annotated[str, "Ticker symbol of the company"],
) -> str:
    """
    Fetch comprehensive earnings data:
    - Upcoming earnings calendar (next date, EPS/revenue estimates)
    - Recent earnings dates with EPS actual vs estimate (surprise %)
    - Quarterly revenue and net income trend
    - Analyst EPS revision trend
    """
    import yfinance as yf
    try:
        t = yf.Ticker(ticker)
        lines = [f"=== Earnings Data for {ticker} ===\n"]

        # ── Upcoming earnings calendar ─────────────────────────
        try:
            cal = t.calendar
            if cal is not None and len(cal) > 0:
                lines.append("## Upcoming Earnings Calendar")
                if hasattr(cal, "to_string"):
                    lines.append(cal.to_string())
                else:
                    lines.append(str(cal))
                lines.append("")
        except Exception:
            pass

        # ── Earnings dates with EPS surprise ──────────────────
        try:
            ed = t.earnings_dates
            if ed is not None and not ed.empty:
                lines.append("## Recent Earnings Dates — EPS Surprise")
                recent = ed.head(8)
                lines.append(recent.to_string())

                # Compute average beat rate
                if "Surprise(%)" in recent.columns:
                    surprises = recent["Surprise(%)"].dropna()
                    if len(surprises) > 0:
                        beat_rate = (surprises > 0).sum() / len(surprises) * 100
                        avg_surprise = surprises.mean()
                        lines.append(
                            f"\nBeat Rate (last {len(surprises)} quarters): {beat_rate:.0f}%"
                            f"  |  Avg Surprise: {avg_surprise:+.1f}%"
                        )
                lines.append("")
        except Exception:
            pass

        # ── Quarterly income highlights ───────────────────────
        try:
            qi = t.quarterly_income_stmt
            if qi is not None and not qi.empty:
                lines.append("## Quarterly Income Highlights (last 4Q)")
                key_rows = [
                    "Total Revenue", "Gross Profit", "Operating Income",
                    "Net Income", "EBITDA", "Basic EPS",
                ]
                found = []
                for row in key_rows:
                    if row in qi.index:
                        found.append(qi.loc[row])
                if found:
                    import pandas as pd
                    df = pd.DataFrame(found)
                    df.columns = [str(c)[:10] for c in df.columns]
                    lines.append(df.iloc[:, :4].to_string())
                lines.append("")
        except Exception:
            pass

        # ── Analyst recommendations ───────────────────────────
        try:
            rec = t.recommendations
            if rec is not None and not rec.empty:
                lines.append("## Analyst Recommendations (recent)")
                lines.append(rec.tail(6).to_string())
                lines.append("")
        except Exception:
            pass

        return "\n".join(lines) if len(lines) > 1 else f"No earnings data available for {ticker}."
    except Exception as e:
        return f"Earnings data fetch error: {e}"
