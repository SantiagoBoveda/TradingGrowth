from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_indicators,
    get_language_instruction,
    get_stock_data,
)
from tradingagents.dataflows.config import get_config


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_stock_data,
            get_indicators,
        ]

        system_message = (
            "You are a senior technical analyst at a top-tier trading desk. Produce an institutional-quality technical analysis report. "
            "Call get_stock_data FIRST to retrieve raw OHLCV data, then call get_indicators for each selected indicator.\n\n"
            "AVAILABLE INDICATORS — choose 8 complementary ones:\n"
            "Moving Averages: close_50_sma (medium-term trend), close_200_sma (long-term/golden cross), close_10_ema (short-term momentum)\n"
            "MACD: macd (momentum), macds (signal line crossovers), macdh (histogram divergence)\n"
            "Momentum: rsi (overbought/oversold + divergence)\n"
            "Volatility: boll (middle band), boll_ub (upper/overbought), boll_lb (lower/oversold), atr (volatility regime, stop-loss sizing)\n"
            "Volume: vwma (volume-weighted trend confirmation)\n\n"
            "MANDATORY ANALYSIS SECTIONS:\n"
            "1. **Trend Analysis**: Primary trend via 50/200 SMA. Price above/below MAs? Golden cross or death cross? Distance from 200 SMA as % (mean-reversion risk).\n"
            "2. **Fibonacci Retracement Levels**: From the OHLCV data, identify the most recent significant swing high and swing low. "
            "Compute and explicitly state: 23.6%, 38.2%, 50%, 61.8%, 78.6% retracement levels with exact prices. "
            "Is price currently at/near a Fibonacci level acting as support or resistance? "
            "Identify Fibonacci extension targets (127.2%, 161.8%) for upside projections.\n"
            "3. **Momentum (RSI & MACD)**: RSI level and trend. Explicitly check for BULLISH DIVERGENCE "
            "(price lower low + RSI higher low = reversal signal) and BEARISH DIVERGENCE "
            "(price higher high + RSI lower high = weakness). MACD histogram expanding or shrinking?\n"
            "4. **Volume Profile (VWMA)**: Is price above or below VWMA? Volume on up-days vs down-days "
            "(buying vs selling pressure). High-volume reversal candles (capitulation or distribution). "
            "Divergence between price trend and volume trend.\n"
            "5. **Volatility Assessment**: Bollinger Band width — expanding (breakout likely) or contracting (consolidation). "
            "ATR: current volatility regime. Price near upper/lower band (mean-reversion setup)?\n"
            "6. **Key Support & Resistance**: 3 support levels + 3 resistance levels using MAs, Fibonacci levels, swing points, and Bollinger Bands.\n"
            "7. **Chart Pattern Recognition**: From OHLCV, identify visible patterns: head & shoulders, double top/bottom, "
            "cup & handle, flags, wedges, triangles. State if pattern is confirmed or forming.\n"
            "8. **Technical Verdict**: BUY / HOLD / SELL with: entry zone, stop-loss level (ATR-based), "
            "Target 1 and Target 2 (Fibonacci extensions or resistance), risk/reward ratio.\n\n"
            "Cite exact prices and numbers throughout. Append a Markdown summary table."
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "market_report": report,
        }

    return market_analyst_node
