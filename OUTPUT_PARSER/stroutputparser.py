from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
      provider="auto"
)

model = ChatHuggingFace(llm=llm)


prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
).invoke({'topic':'blackhole'})


result = model.invoke(prompt1)

prompt2 = PromptTemplate(
    template='Write a summary on the following text. /n {text}',
    input_variables=['text']
).invoke({'text':result.content})


result1 = model.invoke(prompt2)