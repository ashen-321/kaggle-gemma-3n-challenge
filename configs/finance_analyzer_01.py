import sys
import warnings
from crewai import Crew, Agent, Task, LLM, Process
import os
from crewai_tools import ScrapeWebsiteTool, SerperDevTool, WebsiteSearchTool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import TavilySearchResults
from duckduckgo_search import AsyncDDGS
from langchain.tools import tool, Tool
from datetime import datetime

module_paths = ["./", "../scripts", "/home/alfred/utils"]
for module_path in module_paths:
    sys.path.append(os.path.abspath(module_path))
from bedrock import *

#os.environ['OPENAI_MODEL_NAME'] = "gpt-4-turbo"
#os.environ['OPENAI_API_KEY'] = "Your OPEN AI Api Key"
os.environ['SERPER_API_KEY'] = os.getenv('serp_api_token')
os.environ['TAVILY_API_KEY'] = os.getenv('tavily_api_token')



warnings.filterwarnings('ignore')
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()
duck_search_tool = DuckDuckGoSearchRun()
tavily_tool = TavilySearchResults(max_results=5)
web_rag_tool = WebsiteSearchTool()

# Define a new tool that returns the current datetime
def today_date():
    import datetime 
    now = datetime.datetime.now()
    return now.strftime("Today's date: %m/%d/%Y") 
    
@tool("Internet Search Tool")
def internet_search_tool(query: str) -> list:
    """Search Internet for relevant information based on a query."""
    ddgs = AsyncDDGS()
    results = ddgs.text(keywords=query, region='wt-wt', safesearch='moderate', max_results=5)
    return results

# Construct tools
tools = [
    Tool(
        name="Today date",  # Name of the tool
        func=today_date,  # Function that the tool will execute
        # Description of the tool
        description="Useful for when you need to know the today's date",
    ), 
    tavily_tool,
    duck_search_tool,
    web_rag_tool,
]

# CrewAI LLM config uses LiteLLM
llm_haiku = LLM(model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0")
llm_sonnet35 = LLM(model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0")
llm_llama32 = LLM(model="bedrock/meta.llama3-1-70b-instruct-v1:0")
llm_mistral = LLM(model="bedrock/mistral.mistral-large-2407-v1:0")

class finAgents():
    def __init__(self, model_id, stock_selection, initial_capital, investment_duration, risk_tolerance, return_expectation, investor_persona):
        self.stock_selection = stock_selection
        self.initial_capital = initial_capital
        self.investment_duration = investment_duration
        self.risk_tolerance = risk_tolerance
        self.return_expectation = return_expectation
        self.model_id = model_id
        self.investor_persona = investor_persona
        
    def data_analyst_agent(self,model_id, stock_selection, initial_capital, investment_duration, risk_tolerance, return_expectation, investor_persona):
        return  Agent(
            role="Data Analyst",
            goal="Monitor and analyze market data in real-time "
                 f"to identify trends and predict market movements for stocks like ({stock_selection})."
                 f"or stocks in the ({stock_selection})sector ."
                 f"which can achieve the investor's expected return of ({return_expectation})"
                 f"after assessing this investor's persona based on ({investor_persona})"
                 f"based on the investor's available investment amount of ({initial_capital})"
                 f"based on the investor's investment duration of ({investment_duration}) horizon"
                 f"based on the investor's risk tolerance level of ({risk_tolerance})",
            backstory="Specializing in financial markets, this agent "
                      "uses statistical modeling and machine learning "
                      "to provide crucial insights. With a knack for data, "
                      "the Data Analyst Agent is the cornerstone for "
                      "informing trading decisions.",
            allow_delegation=True,
            verbose=True,
            tools=tools,
            llm=LLM(model=model_id),
        )

    def trading_strategy_agent(self, model_id, stock_selection, initial_capital, investment_duration, risk_tolerance, return_expectation, investor_persona):
        return  Agent(
            role="Trading Strategy Developer",
            goal="Develop and test various trading strategies based "
                 "on insights from the Data Analyst Agent."
                 f"Your strategy recommendations should be based on the investor's profile as ({investor_persona})"
                 f"Your strategies should be based on the investment amount ({initial_capital}), investment duration ({investment_duration}),"
                 f"risk tolerance ({risk_tolerance}) and expected investment return ({return_expectation})",
            backstory="Equipped with a deep understanding of financial "
                      "markets and quantitative analysis, this agent "
                      "devises and refines trading strategies. It evaluates "
                      "the performance of different approaches to determine "
                      "the most profitable and risk-averse options.",
            allow_delegation=True,
            tools=tools,
            verbose=True,
            llm=LLM(model=model_id)
        )

    def execution_agent(self, model_id, stock_selection, initial_capital, investment_duration, risk_tolerance, return_expectation):
        return Agent(
            role="Trade Advisor",
            goal="Suggest optimal trade execution strategies "
                 "based on approved trading strategies.",
            backstory="This agent specializes in analyzing the timing, price, "
                      "and logistical details of potential trades. By evaluating "
                      "these factors, it provides well-founded suggestions for "
                      "when and how trades should be executed to maximize "
                      "efficiency and adherence to strategy.",
            allow_delegation=True,
            tools=tools,
            verbose=True,
            llm=LLM(model=model_id)
        )

    def risk_management_agent(self, model_id, stock_selection, initial_capital, investment_duration, risk_tolerance, return_expectation):
        return Agent(
            role="Risk Advisor",
            goal="Evaluate and provide insights on the risks "
                 "associated with potential trading activities."
                 f"Consider user-defined risk tolerance of ({risk_tolerance}). ",
            backstory="Armed with a deep understanding of risk assessment models "
                      "and market dynamics, this agent scrutinizes the potential "
                      "risks of proposed trades. It offers a detailed analysis of "
                      "risk exposure and suggests safeguards to ensure that "
                      "trading activities align with the firm’s risk tolerance.",
            allow_delegation=True,
            tools=tools,
            verbose=True,
            llm=LLM(model=model_id)
        )

    def portfolio_recommend_agent(self, model_id, stock_selection, initial_capital, investment_duration, risk_tolerance, return_expectation):
        return Agent(
            role="Portfolio recommender",
            goal="Assess the overall investment amount, duration, risk tolerance level and return on investiment expection."
                 "Recommend a portfolio to meet those requirements, including stock tickers, amount and time to investion."
                 "You reecommendation should include investment amount, time and actions (i.e. buy, sell ot hold) plus potential transaction/management costs",
            backstory="Armed with a deep understanding of risk assessment models "
                      "and market dynamics, this agent scrutinizes the potential "
                      "risks of proposed trades. It offers a detailed portfolio recommendartion"
                      "to best match the investor's expectations",
            allow_delegation=True,
            tools=tools,
            verbose=True,
            llm=LLM(model=model_id)
        )

class finTasks():
    def __init__(self, stock_selection, initial_capital, investment_duration, risk_tolerance, return_expectation,trading_strategy_preference, news_impact_consideration, investor_persona):
        self.stock_selection = stock_selection
        self.initial_capital = initial_capital
        self.investment_duration = investment_duration
        self.risk_tolerance = risk_tolerance
        self.return_expectation = return_expectation
        self.trading_strategy_preference = trading_strategy_preference
        self.news_impact_consideration = news_impact_consideration
        self.investor_persona = investor_persona

    def data_analysis_task(self, data_analyst_agent, stock_selection):
        return Task(
            description=(
                "Continuously monitor and analyze market data for "
                f"the selected stocks ({stock_selection}). "
                f"or your recommended stocks in ({stock_selection}) sectod"
                "Use statistical modeling and machine learning to "
                "identify trends and predict market movements."
            ),
            expected_output=(
                "Insights and alerts about significant market "
                f"opportunities or threats for stocks ({stock_selection})."
            ),
            agent=data_analyst_agent,
        )

    def strategy_development_task(self, trading_strategy_agent, stock_selection, initial_capital, investment_duration, risk_tolerance, return_expectation, trading_strategy_preference):
        return Task(
            description=(
                "Develop and refine trading strategies based on "
                "the insights from the Data Analyst and "
                f"user-defined risk tolerance of ({risk_tolerance}). "
                f"Consider trading preferences of ({trading_strategy_preference})."
                f"Consider investment duration of ({investment_duration})."
                f"Consider initial investment amount of ({initial_capital})."
                f"Consider initial investment return expectation of ({return_expectation})."
            ),
            expected_output=(
                f"A set of potential trading strategies for stocks {stock_selection} "
                "that align with the user's risk tolerance."
            ),
            agent=trading_strategy_agent,
        )

    def execution_planning_task(self, execution_agent, stock_selection, initial_capital, investment_duration, risk_tolerance, return_expectation, trading_strategy_preference):
        return Task(
            description=(
                "Analyze approved trading strategies to determine the "
                f"best execution methods for stocks ({stock_selection}), "
                "considering current market conditions and optimal pricing."
                f"consider user-defined risk tolerance of ({risk_tolerance}). "
                f"Consider trading preferences of ({trading_strategy_preference})."
                f"Consider investment duration of ({investment_duration})."
                f"Consider initial investment amount of ({initial_capital})."
                f"Consider initial investment return expectation of ({return_expectation})."
            ),
            expected_output=(
                "Detailed execution plans suggesting how and when to "
                "execute trades for the stock portifolio."
            ),
            agent=execution_agent,
        )

    def risk_assessment_task(self, risk_management_agent,stock_selection, risk_tolerance):
        return Task(
            description=(
                "Evaluate the risks associated with the proposed trading "
                f"strategies and execution plans for stocks in ({stock_selection}). "
                f"consider user-defined risk tolerance of ({risk_tolerance}). "
                "Provide a detailed analysis of potential risks, to mitigate below the user-defined risk tolerance of ({risk_tolerance}). "
                "and suggest mitigation strategies."
            ),
            expected_output=(
                "A comprehensive risk analysis report detailing potential "
                "risks and mitigation recommendations for the stock portfolio."
            ),
            agent=risk_management_agent,
        )

    def portfolio_recommend_task(self, portfolio_recommend_agent, stock_selection, initial_capital, investment_duration, risk_tolerance, return_expectation, news_impact_consideration, investor_persona):
        # Task for Risk Advisor Agent: Assess Trading Risks
        return Task(
            description=(
                f"Assess this investor's persona based on ({investor_persona})"
                f"Evaluate the investor's available investment amount of ({initial_capital})"
                f"Based on the preferred stock picks by this investor with ({stock_selection})"
                f"Determine the investor's investment duration of ({investment_duration}) horizon"
                f"Assess the investor's risk tolerance level of ({risk_tolerance})"
                f"Understand the investor's expected return of ({return_expectation})"
                "Conduct thorough analysis of current stock market performance"
                "Identify and interpret market trends across various sectors"
                "Evaluate risk factors associated with different stocks and market segments"
                "Analyze historical and projected returns for potential investment options"
                "Monitor and assess overall market movements and volatility"
                f"If {news_impact_consideration}, Stay informed about Federal Reserve interest rate decisions and their potential impact"
                f"If {news_impact_consideration}, also evaluate geopolitical events and their influence on market stability"
                f"If {news_impact_consideration}, also consider the effects of natural disasters or other unforeseen events on specific industries or the broader market"
            ),
            expected_output=(
                "Develop a diversified portfolio strategy based on the investor's profile and market analysis"
                "Balance short-term and long-term investment opportunities"
                "Incorporate risk management techniques to mitigate potential losses"
                "Optimize asset allocation to meet the investor's return expectations within their risk tolerance"
                "Create a clear and comprehensive stock portfolio recommendation"
                "Provide rationale for each selected stocks or investment vehicle"
                "Explain how the recommended portfolio aligns with the investor's goals and risk profile"
                "Outline potential scenarios for portfolio performance under various market conditions"
                "Regularly review and assess the performance of recommended portfolios"
                "Stay alert to changes in market conditions or external factors that may impact the portfolio"
                "Provide timely advice on rebalancing or adjusting the portfolio as needed"
                "Communicate updates and recommendations to the investor promptly"
                "Explain complex market concepts and investment strategies in clear, understandable terms"
                "Keep investors informed about market trends and potential impacts on their portfolios"
                "Provide guidance on how to interpret portfolio performance and market events"
            ),
            agent=portfolio_recommend_agent,
        )

class stockCrew():
  def __init__(self, inputs: dict,
            model_id: str='bedrock/anthropic.claude-3-haiku-20240307-v1:0', manager_model_id: str=llm_sonnet35,
            embedding_provider: str='aws_bedrock', embedding_model_id: str='amazon.titan-embed-text-v2:0', embedding_dimensions: int=1024,):
    self.inputs = inputs
    self.model_id = model_id
    self.manager_model_id = manager_model_id
    self.embedding_provider = embedding_provider
    self.embedding_model_id = embedding_model_id
    self.embedding_provider = embedding_provider
    self.embedding_dimensions = embedding_dimensions
    self.stock_selection = inputs['stock_selection']
    self.initial_capital =  inputs['initial_capital']
    self.investment_duration = inputs['investment_duration']
    self.risk_tolerance = inputs['risk_tolerance']
    self.return_expectation = inputs['return_expectation']
    self.trading_strategy_preference = inputs['trading_strategy_preference']
    self.news_impact_consideration  = inputs['news_impact_consideration']
    self.investor_persona = inputs['investor_persona']

  def run(self):
    agents = finAgents(self.model_id, self.stock_selection, self.initial_capital, self.investment_duration, self.risk_tolerance, self.return_expectation, self.investor_persona)
    tasks = finTasks(self.stock_selection, self.initial_capital, self.investment_duration, self.risk_tolerance, self.return_expectation, self.trading_strategy_preference, self.news_impact_consideration, self.investor_persona)

    data_analyst_agent = agents.data_analyst_agent(self.model_id, self.stock_selection, self.initial_capital, self.investment_duration, self.risk_tolerance, self.return_expectation, self.investor_persona)
    trading_strategy_agent = agents.trading_strategy_agent(self.model_id, self.stock_selection, self.initial_capital, self.investment_duration, self.risk_tolerance, self.return_expectation, self.investor_persona)
    execution_agent = agents.execution_agent(self.model_id, self.stock_selection, self.initial_capital, self.investment_duration, self.risk_tolerance, self.return_expectation)
    risk_management_agent = agents.risk_management_agent(self.model_id, self.stock_selection, self.initial_capital, self.investment_duration, self.risk_tolerance, self.return_expectation)
    portfolio_recommend_agent = agents.portfolio_recommend_agent(self.model_id, self.stock_selection, self.initial_capital, self.investment_duration, self.risk_tolerance, self.return_expectation)

    data_analysis_task = tasks.data_analysis_task(data_analyst_agent, self.stock_selection)
    strategy_development_task = tasks.strategy_development_task(trading_strategy_agent, self.stock_selection, self.initial_capital, self.investment_duration, self.risk_tolerance, self.return_expectation, self.trading_strategy_preference)
    execution_planning_task = tasks.execution_planning_task(execution_agent, self.stock_selection, self.initial_capital, self.investment_duration, self.risk_tolerance, self.return_expectation, self.trading_strategy_preference)
    risk_assessment_task = tasks.risk_assessment_task(risk_management_agent, self.stock_selection, self.risk_tolerance)
    portfolio_recommend_task = tasks.portfolio_recommend_task(portfolio_recommend_agent, self.stock_selection, self.initial_capital, self.investment_duration, self.risk_tolerance, self.return_expectation, self.news_impact_consideration, self.investor_persona)

    # define the crew with agent and tasks
    financial_trading_crew = Crew (
        agents=[data_analyst_agent, 
                trading_strategy_agent, 
                execution_agent, 
                risk_management_agent,
                portfolio_recommend_agent],
        
        tasks=[data_analysis_task, 
               #strategy_development_task, 
               #execution_planning_task, 
               #risk_assessment_task,
               portfolio_recommend_task],
        #task_result = data_analyst_agent.execute_task(data_analysis_task),
        verbose=True,
        process=Process.hierarchical,
        respect_context_window=True,
        planning=True,  # Enable planning feature for pre-execution strategy
        manager_llm=LLM(model=self.manager_model_id), # or manager_agent=None
        memory=True,
        embedder={
            "provider": self.embedding_provider,
            #"provider": "huggingface",
            "config":{
                "model": self.embedding_model_id,
                "vector_dimension": self.embedding_dimensions
                #"model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            }
        }
    )

    result = financial_trading_crew.kickoff(inputs=self.inputs)
    return result

if __name__ == "__main__":
    financial_trading_inputs = {
        'stock_selection': 'NVDA, META',
        'initial_capital': 'USD $10,000',
        'risk_tolerance': 'Moderate',
        'investment_duration': '5 years',
        'return_expectation': '8% annual',
        'trading_strategy_preference': "Monthly Trading",
        'news_impact_consideration': True,
        'investor_persona': "I am a 30-year old engineer who has a stable job and am looking to retire at 50 years old with comfortable life style afterward."
    }
    stock_crew = stockCrew(inputs=financial_trading_inputs)
    result = stock_crew.run()
