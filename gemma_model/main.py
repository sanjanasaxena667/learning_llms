import os
from dotenv import load_dotenv

from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true";
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

## prompt template
prompt=ChatPromptTemplate.from_messages(
  [
    ("system","you are a helpful assistent. please respond to the question asked."),
    ("user","question:{question}")
  ]
)

## streamlit framework

st.title("langchain demo with gemma2:2b")
input_text=st.text_input("what question you have in your mind?")

## call gemma2:2b model
llm = Ollama(model="gemma2:2b")

output_parser=StrOutputParser()

chain=prompt|llm|output_parser

if input_text:
  response = chain.invoke({"question": input_text})
  st.write(response)

