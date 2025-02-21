from phi.agent import Agent
import phi.api
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import os
from dotenv import load_dotenv

import phi
from phi.playground import Playground, serve_playground_app
load_dotenv()

phi.api=os.getenv("PHI_API_KEY")

## fiancial agent
finance_agent = Agent(  
    name="Finance_AI_Agent",
    model= Groq(id="llama3-groq-70b-8192-tool-use-pr"),
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
    model = Groq(id="llama3-groq-70b-8192-tool-use-pr"),
    tools=[DuckDuckGo()],
    instructions=["always include sources"],
    show_tool_calls=True,
    markdown=True
)

app = Playground(agents=["finance_agent", "web_search_agent"]).get_app()

if __name__=="__main__":
    serve_playground_app("playground:app", reload=True)