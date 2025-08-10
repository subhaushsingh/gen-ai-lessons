from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


load_dotenv()


#model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

result = llm.invoke('What is the capital of india?')

print(result)
print('------------------------')
print(result.content)



