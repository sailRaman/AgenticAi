from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

## fiancial agent
finance_agent = Agent(  
    name="Finance_AI_Agent",
    model= Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,company_news=True)],
    show_tool_calls=True,
    description="You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
    instructions=["Format your response using markdown and use tables to display data where possible."],
    markdown=True
)


## websearch agent
web_search_agent = Agent(
    name="Web_Search_Agent",
    role = "Search web for information",
    model = Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["always include sources"],
    show_tool_calls=True,
    markdown=True
)
multi_ai_agent = Agent(team=[web_search_agent, finance_agent],
                       model = Groq(id="llama-3.3-70b-versatile"),
                       instructions=["Always include sources","use tables to display data"],
                       show_tool_calls=True,
                       markdown=True)

multi_ai_agent.print_response("Summarise analyst recommendations and share the latest news for NVDA", markdown=True)