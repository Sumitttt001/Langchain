from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

template = "Write a {tone} email to {company} expressing interest in the {position} position," \
"mentioning {skills} as a key strength . keep it to 4 line max"

prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({
    "tone": "professional",
    "company": "Google",
    "position": "AI Engineer",
    "skills": "machine learning, natural language processing, and cloud computing"
})

result = llm.invoke(prompt)

print(result.content)