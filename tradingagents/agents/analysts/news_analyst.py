from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_global_news,
    get_language_instruction,
    get_news,
)
from tradingagents.agents.utils.news_data_tools import get_newsapi_news


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_news,
            get_global_news,
            get_newsapi_news,
        ]

        system_message = (
            "You are a senior news analyst and researcher. Your goal is to produce a comprehensive, "
            "actionable news report covering company-specific developments and macroeconomic context. "
            "Use ALL available tools:\n"
            "- `get_news(query, start_date, end_date)` — for company-specific news via yfinance\n"
            "- `get_newsapi_news(query, from_date, to_date)` — for broader news coverage via NewsAPI (use ticker + company name as query)\n"
            "- `get_global_news(curr_date, look_back_days)` — for macro/global market news\n\n"
            "Your report MUST cover:\n"
            "1. **Company News**: Product launches, earnings previews/results, management changes, M&A rumors, "
            "regulatory news, legal issues, analyst upgrades/downgrades.\n"
            "2. **Industry/Sector News**: Competitor moves, sector trends, supply chain developments, "
            "regulatory changes affecting the industry.\n"
            "3. **Macro News**: Fed decisions, inflation data, employment reports, geopolitical events "
            "that could impact this stock.\n"
            "4. **Sentiment Assessment**: Is the overall news flow POSITIVE, NEGATIVE, or MIXED? "
            "Are there any upcoming catalysts (earnings, product launches, regulatory decisions)?\n"
            "5. **News-Based Signal**: Based purely on news, is the tone BUY / HOLD / SELL? "
            "Identify the 3 most important news items and their likely market impact.\n\n"
            "Be specific — cite headlines, dates, and sources. "
            "Append a Markdown table summarizing key news items."
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
            "news_report": report,
        }

    return news_analyst_node
