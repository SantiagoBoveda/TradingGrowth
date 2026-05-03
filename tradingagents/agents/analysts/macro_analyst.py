from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_global_news,
    get_language_instruction,
)
from tradingagents.agents.utils.macro_tools import get_macro_indicators, get_sector_performance


def create_macro_analyst(llm):
    def macro_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_macro_indicators,
            get_sector_performance,
            get_global_news,
        ]

        system_message = (
            "You are a senior macro analyst at a top investment bank. Your role is to assess the macroeconomic "
            "environment and its specific implications for the stock being analyzed. "
            "Use ALL tools: `get_macro_indicators` for VIX, rates, DXY, and major indices; "
            "`get_sector_performance` for sector rotation analysis; "
            "`get_global_news` for macro-relevant news (Fed, inflation, geopolitics, earnings season). "
            "\n\nYour report MUST cover ALL of these sections:\n"
            "1. **Risk Environment (VIX)**: Current VIX level and regime (complacent <15, normal 15-25, fearful 25-35, crisis >35). "
            "Is the market in risk-on or risk-off mode? Implications for the stock.\n"
            "2. **Interest Rate Environment**: 10yr yield direction and level. Fed stance (hawkish/dovish/neutral). "
            "Impact on valuations — especially for growth stocks (higher rates = higher discount rate = lower multiples).\n"
            "3. **USD Strength (DXY)**: Is the dollar strengthening or weakening? "
            "Implications for multinationals (strong USD = revenue headwind), commodities, and EM exposure.\n"
            "4. **Equity Market Breadth**: S&P 500 and NASDAQ trend (30-day). "
            "Is this a broad rally or narrow? Risk of mean reversion?\n"
            "5. **Sector Rotation**: Which sectors are leading/lagging over the past month? "
            "Is the target company's sector benefiting from rotation or facing outflows?\n"
            "6. **Macro Tailwinds & Headwinds for this stock**: "
            "Be SPECIFIC — how does the current macro backdrop directly affect THIS company "
            "(e.g., high rates hurt fintech growth stocks; strong USD hurts Apple's international revenue; "
            "energy bull cycle helps CVX).\n"
            "7. **Macro Catalyst Watch**: Any upcoming macro events (Fed meeting, CPI, jobs report, earnings season) "
            "that could be catalysts for a move.\n"
            "8. **Macro Verdict**: BULLISH / NEUTRAL / BEARISH macro backdrop for this stock, "
            "with a clear 1-3 sentence summary.\n"
            "\nBe specific — cite exact levels and percentage changes. "
            "Append a Markdown summary table of all macro indicators at the end."
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages([
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
        ])

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
            "macro_report": report,
        }

    return macro_analyst_node
