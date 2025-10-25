from json import tool
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
import datetime
from langchain.agents import tool

load_dotenv()

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """Returns the current date and time in the specified format."""
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

llm = GoogleGenerativeAI(model = "gemini-2.0-flash")

query = "What is the current time in london? (you are n india) . Just show the current time and not a date."

Prompt_template = hub.pull("hwchase17/react")

tools = [get_system_time]
agent = create_react_agent(llm, tools, Prompt_template)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({"input": query})