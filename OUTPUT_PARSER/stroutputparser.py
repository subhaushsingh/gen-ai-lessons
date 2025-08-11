from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
      provider="auto"
)

model = ChatHuggingFace(llm=llm)


template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)#.invoke({'topic':'blackhole'})


#result = model.invoke(prompt1)

template2 = PromptTemplate(
    template='Write a summary on the following text. /n {text}',
    input_variables=['text']
)#.invoke({'text':result.content})

#result1 = model.invoke(prompt2)



parser = StrOutputParser()


chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'blackhole'})

print(result)
