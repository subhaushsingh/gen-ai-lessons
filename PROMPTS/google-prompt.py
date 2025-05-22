from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import streamlit as st

load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')


st.header("Research Tool")
paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template = PromptTemplate.from_template( """
You are an expert in reading and interpreting academic research papers.

Your task is to explain the core ideas of the research paper titled: "{paper_input}"

Please provide the explanation in a {style_input} style.
- Possible styles: simple and beginner-friendly, technical and academic, or creative and engaging.

Keep the explanation {length_input} in length.
- Use "short" for 2-3 sentences, "medium" for a paragraph, and "detailed" for an in-depth summary with examples if needed.

Begin your explanation now:
""")


prompt = template.format(
    paper_input= paper_input,
    style_input=style_input,
    length_input= length_input
)

if st.button('summarize'):
    result = model.invoke(prompt)
    st.write(result.content)