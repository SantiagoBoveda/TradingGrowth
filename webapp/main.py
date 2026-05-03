"""
TradingAgents Web App — FastAPI Backend
"""
import os, sys, json, re, time, sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# ── paths ──────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

app = FastAPI(title="TradingAgents")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── DB ─────────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent / "history.db"

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker      TEXT NOT NULL,
                company_name TEXT,
                sector      TEXT,
                industry    TEXT,
                date        TEXT NOT NULL,
                price       REAL,
                decision    TEXT,
                scores      TEXT,
                reports     TEXT,
                trader_plan TEXT,
                analysts    TEXT,
                created_at  TEXT DEFAULT (datetime('now'))
            )
        """)
        con.commit()

init_db()

@contextmanager
def get_db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
    finally:
        con.close()

def save_analysis(data: dict, analysts: list):
    with get_db() as con:
        con.execute("""
            INSERT INTO analyses
              (ticker, company_name, sector, industry, date, price,
               decision, scores, reports, trader_plan, analysts)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            data["ticker"],
            data.get("company_name"),
            data.get("sector"),
            data.get("industry"),
            data["date"],
            data.get("price"),
            data["decision"],
            json.dumps(data.get("scores", {})),
            json.dumps({k: v[:800] for k, v in data.get("reports", {}).items()}),
            (data.get("trader_plan") or "")[:1200],
            json.dumps(analysts),
        ))
        con.commit()

# ── models ─────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    ticker: str
    date: str
    analysts: List[str] = ["market", "news", "fundamentals", "social"]
    debate_rounds: int = 1

class BacktestRequest(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    freq_days: int = 30
    holding_days: int = 5

# ── helpers ────────────────────────────────────────────────────
def score_dimensions(reports: dict) -> dict:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        key = os.getenv("GOOGLE_API_KEY")
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=key, temperature=0)

        combined = "\n\n".join(
            f"## {k.upper()}\n{v[:600]}"
            for k, v in reports.items() if v and len(v) > 30
        )

        prompt = (
            "Based on these analyst reports, score the company 0-10 on each dimension.\n\n"
            f"{combined[:2500]}\n\n"
            "Rules:\n"
            "- valuation: 10=extremely cheap, 0=very overvalued\n"
            "- future:    10=excellent growth prospects, 0=declining\n"
            "- past:      10=strong historical performance, 0=very poor\n"
            "- health:    10=fortress balance sheet, 0=insolvency risk\n"
            "- momentum:  10=strong bullish momentum, 0=strong bearish\n\n"
            "Return ONLY JSON, no markdown, no explanation:\n"
            '{"valuation":5,"future":5,"past":5,"health":5,"momentum":5}'
        )

        resp = llm.invoke(prompt)
        m = re.search(r'\{[^}]+\}', resp.content, re.DOTALL)
        if m:
            raw = json.loads(m.group())
            keys = ["valuation", "future", "past", "health", "momentum"]
            return {k: max(0, min(10, int(raw.get(k, 5)))) for k in keys}
    except Exception as e:
        print(f"[score] {e}")
    return {"valuation": 5, "future": 5, "past": 5, "health": 5, "momentum": 5}


def get_stock_price(ticker: str, date: str):
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        h = t.history(start=date, period="2d")
        price = float(h["Close"].iloc[0]) if len(h) > 0 else None
        info  = t.info
        return price, info.get("longName", ticker), info.get("sector", ""), info.get("industry", "")
    except:
        return None, ticker, "", ""


def calc_return(ticker: str, trade_date: str, holding_days: int):
    try:
        import yfinance as yf
        start = datetime.strptime(trade_date, "%Y-%m-%d")
        end   = (start + timedelta(days=holding_days + 7)).strftime("%Y-%m-%d")
        stock = yf.Ticker(ticker).history(start=trade_date, end=end)
        spy   = yf.Ticker("SPY").history(start=trade_date, end=end)
        if len(stock) < 2 or len(spy) < 2:
            return None, None
        n     = min(holding_days, len(stock) - 1, len(spy) - 1)
        raw   = float((stock["Close"].iloc[n] - stock["Close"].iloc[0]) / stock["Close"].iloc[0])
        bench = float((spy["Close"].iloc[n]   - spy["Close"].iloc[0])   / spy["Close"].iloc[0])
        return raw, raw - bench
    except:
        return None, None


def decision_to_position(decision: str) -> int:
    d = decision.lower()
    if any(x in d for x in ["buy", "overweight", "strong buy"]):  return  1
    if any(x in d for x in ["sell", "underweight", "strong sell"]): return -1
    return 0


# ── routes ─────────────────────────────────────────────────────
@app.get("/")
def root():
    return FileResponse(Path(__file__).parent / "index.html")


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    try:
        config = DEFAULT_CONFIG.copy()
        config["max_debate_rounds"]       = req.debate_rounds
        config["max_risk_discuss_rounds"] = req.debate_rounds

        ta = TradingAgentsGraph(
            selected_analysts=req.analysts,
            config=config,
            debug=False,
        )

        state, decision = ta.propagate(req.ticker.upper(), req.date)

        reports = {
            k.replace("_report", ""): state.get(k, "")
            for k in ["market_report", "fundamentals_report", "sentiment_report", "news_report"]
            if state.get(k) and len(state.get(k, "")) > 20
        }

        scores = score_dimensions(reports)
        price, name, sector, industry = get_stock_price(req.ticker.upper(), req.date)

        result = {
            "ticker":       req.ticker.upper(),
            "company_name": name,
            "sector":       sector,
            "industry":     industry,
            "date":         req.date,
            "price":        price,
            "decision":     decision,
            "scores":       scores,
            "reports":      reports,
            "trader_plan":  state.get("trader_investment_plan", ""),
        }

        save_analysis(result, req.analysts)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history")
def get_history(ticker: Optional[str] = None, limit: int = 50):
    with get_db() as con:
        if ticker:
            rows = con.execute(
                "SELECT * FROM analyses WHERE ticker=? ORDER BY created_at DESC LIMIT ?",
                (ticker.upper(), limit)
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT * FROM analyses ORDER BY created_at DESC LIMIT ?",
                (limit,)
            ).fetchall()

    result = []
    for r in rows:
        item = dict(r)
        item["scores"]  = json.loads(item["scores"]  or "{}")
        item["reports"] = json.loads(item["reports"] or "{}")
        item["analysts"]= json.loads(item["analysts"] or "[]")
        result.append(item)
    return result


@app.get("/api/history/tickers")
def get_tickers():
    with get_db() as con:
        rows = con.execute(
            "SELECT ticker, company_name, COUNT(*) as count, MAX(created_at) as last_analysis "
            "FROM analyses GROUP BY ticker ORDER BY last_analysis DESC"
        ).fetchall()
    return [dict(r) for r in rows]


@app.delete("/api/history/{analysis_id}")
def delete_analysis(analysis_id: int):
    with get_db() as con:
        con.execute("DELETE FROM analyses WHERE id=?", (analysis_id,))
        con.commit()
    return {"ok": True}


@app.post("/api/backtest")
def backtest(req: BacktestRequest):
    dates, current = [], datetime.strptime(req.start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(req.end_date, "%Y-%m-%d")
    while current <= end_dt:
        if current.weekday() < 5:
            dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=req.freq_days)

    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"] = config["max_risk_discuss_rounds"] = 1

    ta = TradingAgentsGraph(
        selected_analysts=["market", "fundamentals"],
        config=config, debug=False,
    )

    results = []
    for date in dates:
        try:
            _, decision = ta.propagate(req.ticker.upper(), date)
            pos         = decision_to_position(decision)
            raw, alpha  = calc_return(req.ticker.upper(), date, req.holding_days)
            pnl         = float(raw * pos) if raw is not None else None
            results.append({
                "date": date, "decision": decision, "position": pos,
                "raw_ret": round(float(raw)  * 100, 2) if raw   is not None else None,
                "alpha":   round(float(alpha)* 100, 2) if alpha is not None else None,
                "pnl":     round(pnl         * 100, 2) if pnl   is not None else None,
            })
            time.sleep(1)
        except Exception as e:
            results.append({"date": date, "decision": "ERROR", "position": 0,
                            "raw_ret": None, "alpha": None, "pnl": None, "error": str(e)})

    valid = [r for r in results if r["pnl"] is not None]
    summary = {}
    if valid:
        cum_pnl = 0
        for r in results:
            if r["pnl"] is not None:
                cum_pnl += r["pnl"]
                r["cum_pnl"] = round(cum_pnl, 2)

        total_pnl   = sum(r["pnl"]   for r in valid)
        total_alpha = sum(r["alpha"] for r in valid)
        wins        = sum(1 for r in valid if r["pnl"] > 0)
        summary = {
            "total_pnl":   round(total_pnl,   2),
            "total_alpha": round(total_alpha, 2),
            "win_rate":    round(wins / len(valid) * 100, 1),
            "trades":      len(valid),
            "buys":        sum(1 for r in valid if r["position"] ==  1),
            "holds":       sum(1 for r in valid if r["position"] ==  0),
            "sells":       sum(1 for r in valid if r["position"] == -1),
        }

    return {"ticker": req.ticker.upper(), "results": results, "summary": summary}


@app.get("/api/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
