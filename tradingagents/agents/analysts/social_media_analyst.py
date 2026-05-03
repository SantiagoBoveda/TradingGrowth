from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
    get_news,
)
from tradingagents.agents.utils.news_data_tools import (
    get_stocktwits_sentiment,
    get_reddit_sentiment,
)


def create_social_media_analyst(llm):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_stocktwits_sentiment,
            get_reddit_sentiment,
            get_news,
        ]

        system_message = (
            "You are a senior social media and sentiment analyst. Your role is to gauge retail investor "
            "sentiment and identify emerging narratives before they reach mainstream financial media. "
            "Use ALL available tools:\n"
            "- `get_stocktwits_sentiment(ticker)` — real-time trader sentiment from StockTwits (bullish/bearish ratio + recent posts)\n"
            "- `get_reddit_sentiment(ticker, lookback_days)` — Reddit discussions from r/wallstreetbets, r/stocks, r/investing\n"
            "- `get_news(query, start_date, end_date)` — company news and social media coverage\n\n"
            "Your report MUST cover:\n"
            "1. **StockTwits Sentiment**: Bullish vs bearish ratio. Is the community predominantly bullish or bearish? "
            "Any notable price targets or catalysts mentioned? Sentiment extreme (contrarian signal)?\n"
            "2. **Reddit Sentiment**: Top posts by upvotes. Is this stock being heavily discussed (meme potential)? "
            "What is the dominant narrative — fundamental bullish thesis, short squeeze setup, or bearish concerns? "
            "Any viral posts that could drive retail FOMO or panic selling?\n"
            "3. **Sentiment Trends**: Is sentiment improving or deteriorating over the past week? "
            "Disconnect between social sentiment and fundamentals (contrarian opportunity)?\n"
            "4. **Retail vs Institutional**: Are retail investors positioned opposite to what fundamentals suggest? "
            "Crowded long or short by retail (contrarian signal)?\n"
            "5. **Viral Catalysts**: Any memes, viral posts, or social media campaigns that could cause "
            "outsized price movement regardless of fundamentals (WSB-style short squeezes, etc.).\n"
            "6. **Social Sentiment Signal**: Based on social data, is the crowd signal BUY / HOLD / SELL? "
            "Note: contrarian interpretation often applies when sentiment is at extremes.\n\n"
            "Be specific — cite actual post content, exact bullish/bearish counts, and subreddit sources. "
            "Append a Markdown summary table."
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
            "sentiment_report": report,
        }

    return social_media_analyst_node
