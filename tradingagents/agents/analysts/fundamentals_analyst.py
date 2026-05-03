from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
    get_insider_transactions,
    get_language_instruction,
)
from tradingagents.dataflows.config import get_config


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
            get_insider_transactions,
        ]

        system_message = (
            "You are a senior fundamental analyst at a top-tier investment bank. Your task is to produce a deeply rigorous, institutional-quality fundamental analysis report on a company. "
            "Use ALL available tools to gather the most complete picture possible: `get_fundamentals` for valuation multiples and key ratios, `get_balance_sheet` for both annual and quarterly data, `get_cashflow` for both annual and quarterly data, `get_income_statement` for both annual and quarterly data, and `get_insider_transactions` for insider buying/selling signals. "
            "Your report MUST cover the following sections in depth: "
            "1. **Valuation Analysis**: P/E (trailing & forward), PEG, P/B, EV/EBITDA, P/FCF — compare to sector peers and historical averages. Flag if over/undervalued. "
            "2. **Profitability & Margins**: Gross margin, operating margin, net margin, ROE, ROA, ROIC trends over the last 4-8 quarters. Identify deterioration or improvement. "
            "3. **Revenue & Earnings Quality**: YoY and QoQ revenue growth, earnings consistency, one-time items, earnings manipulation risk (Beneish M-Score indicators if possible). "
            "4. **Balance Sheet Strength**: Debt/Equity, Net Debt, Current Ratio, Interest Coverage Ratio. Assess financial leverage and solvency risk. "
            "5. **Cash Flow Analysis**: Operating cash flow vs net income (accruals ratio), CapEx intensity, Free Cash Flow generation, FCF yield, dividend sustainability (FCF payout ratio). "
            "6. **Capital Allocation**: Dividends, buybacks, acquisitions — are they value-accretive or destroying capital? Is the company funding shareholder returns with debt? "
            "7. **Insider Activity**: Recent insider transactions — heavy selling is a red flag, buying is a bullish signal. Analyze patterns and amounts. "
            "8. **Red Flags & Risks**: Identify any accounting concerns, unsustainable trends, debt maturity walls, or structural business deterioration. "
            "9. **Bull vs Bear Case**: Present both sides with specific data points. "
            "10. **Final Fundamental Verdict**: Clear BUY / HOLD / SELL based purely on fundamentals with a target price range if possible. "
            "Use BOTH annual AND quarterly data for trends. Be specific — cite exact numbers, percentages, and dates. Do not be vague. "
            "Append a comprehensive Markdown summary table at the end with all key metrics."
            + get_language_instruction(),
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
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
