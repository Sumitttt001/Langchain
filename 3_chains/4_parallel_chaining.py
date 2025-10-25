from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Prompt to get summary
summary_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie critic."),
        ("human", "provide a brief summary of a movie {movie_name}."),
    ]
)

# Template for plot analysis
def analyze_plot(plot):
    plot_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a movie critic."),
            ("human", "Analyze the plot: {plot}. what are its strengths and weaknesses?"),
        ]
    )
    return plot_template.format_prompt(plot=plot)

# Template for character analysis
def analyze_characters(characters):
    character_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a movie critic."),
            ("human", "Analyze the characters: {characters}. what are their strengths and weaknesses?"),
        ]
    )
    return character_template.format_prompt(characters=characters)

# Combine results
def combine_verdict(plot_analysis, character_analysis):
    return f"Plot Analysis:\n{plot_analysis}\n\nCharacter Analysis:\n{character_analysis}"

# Plot analysis branch
plot_branch_chain = (
    RunnableLambda(lambda x: analyze_plot(x["summary"])) | model | StrOutputParser()
)

# Character analysis branch
character_branch_chain = (
    RunnableLambda(lambda x: analyze_characters(x["summary"])) | model | StrOutputParser()
)

# Full chain
chain = (
    summary_template
    | model
    | StrOutputParser()
    | RunnableLambda(lambda summary: {"summary": summary})
    | RunnableParallel({"plot": plot_branch_chain, "characters": character_branch_chain})
    | RunnableLambda(lambda x: combine_verdict(x["plot"], x["characters"]))
)

# Run it
result = chain.invoke({"movie_name": "Inception"})
print(result)
