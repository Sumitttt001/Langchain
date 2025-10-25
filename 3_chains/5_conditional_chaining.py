from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableBranch

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are helpful assistant."),
        ("human", 
         "Generate a feedback note for the positive feedback : {feedback}.")
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are helpful assistant."),
        ("human", 
         "Generate a response addressing this negative feedback : {feedback}.")
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are helpful assistant."),
        ("human", 
         "Generate a response for the more details for this neutral feedback : {feedback}.")
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are helpful assistant."),
        ("human", 
         "generate a message to escalate this feedback to a human agent : {feedback}.")
    ]
)

# Define the feedback classification template
classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are helpful assistant."),
        ("human", 
         "classify the sentiments of this feedback as positive, negative, neutral or escalated: {feedback}.")
    ]
)

# Define the runnable branches for handeling feedback
branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "negative" in x,
        negative_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "neutral" in x,
        neutral_feedback_template | model | StrOutputParser()
    ),
    (
        escalate_feedback_template | model | StrOutputParser()
    )
)

classification_chain = classification_template | model | StrOutputParser()

chain = classification_chain | branches

review = "The product is terrible.It broke after just one use and the quality is very poor."
result = chain.invoke({"feedback": review})

print(result)