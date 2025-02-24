import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Function to extract text from a webpage
def extract_text(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        return "\n".join([p.get_text() for p in paragraphs]) if paragraphs else "No content found on the page."
    except requests.exceptions.RequestException as e:
        return f"Error fetching content: {str(e)}"

# Function to answer questions using LangChain
def answer_question(context, question):
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    template = """You are an AI assistant. Answer based on the given webpage content ONLY.
    If the answer isn't in the content, say "I don't know." 

    Context: {context}
    Question: {question}
    Answer:"""
    
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    chain = LLMChain(llm=llm, prompt=prompt)
    
    return chain.run({"context": context, "question": question})

# Streamlit UI
st.title("🔍 Web Content Q&A Tool")

url = st.text_input("Enter a Website URL:")
if st.button("Extract Content"):
    content = extract_text(url)
    st.session_state['content'] = content
    st.success("Content Extracted Successfully! ✅")

question = st.text_input("Ask a Question:")
if st.button("Get Answer"):
    if 'content' in st.session_state and st.session_state['content']:
        answer = answer_question(st.session_state['content'], question)
        st.write(f"**Answer:** {answer}")
    else:
        st.warning("Please extract content first before asking a question.")
