from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
    get_news,
)
from tradingagents.agents.utils.earnings_tools import get_earnings_data


def create_earnings_analyst(llm):
    def earnings_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_earnings_data,
            get_news,
        ]

        system_message = (
            "You are a senior earnings analyst specializing in earnings quality, surprises, and guidance analysis. "
            "Use `get_earnings_data` to fetch the earnings calendar, EPS surprise history, and quarterly financials. "
            "Use `get_news` to find recent analyst upgrades/downgrades, earnings previews, and guidance revisions. "
            "\n\nYour report MUST cover ALL of these sections:\n"
            "1. **Upcoming Earnings**: Next earnings date, consensus EPS estimate, consensus revenue estimate. "
            "Expected post-earnings move (options-implied if available). Is earnings a near-term catalyst?\n"
            "2. **EPS Surprise History (last 8 quarters)**: Did the company beat or miss? By how much? "
            "Beat rate (%), average surprise magnitude. Is management sandbagging guidance?\n"
            "3. **Revenue vs EPS Quality**: Is EPS growth driven by share buybacks or actual revenue/margin expansion? "
            "Watch for EPS beats on revenue misses — low quality earnings.\n"
            "4. **Guidance Analysis**: Last quarter's guidance — was it raised, maintained, or lowered? "
            "Does management have a track record of beating their own guidance?\n"
            "5. **Analyst Estimate Revisions**: Have estimates been revised up or down recently? "
            "Revision momentum (positive revisions = positive catalyst).\n"
            "6. **Earnings Risk Factors**: Upcoming cost pressures, FX headwinds, supply chain issues, "
            "competitive threats that could cause an earnings miss.\n"
            "7. **Earnings Season Context**: How is the broader sector performing this earnings season? "
            "Are sector peers beating or missing?\n"
            "8. **Earnings Verdict**: Is the earnings outlook a BUY / HOLD / SELL catalyst? "
            "Expected direction of post-earnings move with probability assessment.\n"
            "\nCite exact EPS numbers, percentages, and dates. "
            "Append a Markdown table of the last 8 quarters showing: Date | EPS Est | EPS Actual | Surprise% | Revenue."
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
            "earnings_report": report,
        }

    return earnings_analyst_node
