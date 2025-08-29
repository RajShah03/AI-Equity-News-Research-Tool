import os
import requests
from bs4 import BeautifulSoup
import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA

# -------------------------
# Scrapers for multiple sites
# -------------------------
def scrape_moneycontrol():
    url = "https://www.moneycontrol.com/news/business/"
    res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(res.text, "html.parser")
    links = [a["href"] for a in soup.select("a") if a.get("href") and "/news/" in a["href"]]
    articles = []
    for link in links[:5]:
        try:
            art_res = requests.get(link, headers={"User-Agent": "Mozilla/5.0"})
            art_soup = BeautifulSoup(art_res.text, "html.parser")
            text = " ".join([p.get_text() for p in art_soup.find_all("p")])
            if text:
                articles.append(text)
        except:
            pass
    return articles

def scrape_economictimes():
    url = "https://economictimes.indiatimes.com/markets"
    res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(res.text, "html.parser")
    links = [a["href"] for a in soup.select("a") if a.get("href") and "/markets/" in a["href"]]
    articles = []
    for link in links[:5]:
        if not link.startswith("http"):
            link = "https://economictimes.indiatimes.com" + link
        try:
            art_res = requests.get(link, headers={"User-Agent": "Mozilla/5.0"})
            art_soup = BeautifulSoup(art_res.text, "html.parser")
            text = " ".join([p.get_text() for p in art_soup.find_all("p")])
            if text:
                articles.append(text)
        except:
            pass
    return articles

def scrape_businessstandard():
    url = "https://www.business-standard.com/category/economy-policy-news-1010101.htm"
    res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(res.text, "html.parser")
    links = [a["href"] for a in soup.select("a") if a.get("href") and "/article/" in a["href"]]
    articles = []
    for link in links[:5]:
        if not link.startswith("http"):
            link = "https://www.business-standard.com" + link
        try:
            art_res = requests.get(link, headers={"User-Agent": "Mozilla/5.0"})
            art_soup = BeautifulSoup(art_res.text, "html.parser")
            text = " ".join([p.get_text() for p in art_soup.find_all("p")])
            if text:
                articles.append(text)
        except:
            pass
    return articles

# -------------------------
# Combine scrapers + prepare docs
# -------------------------
def scrape_and_prepare_docs():
    articles = []
    articles.extend(scrape_moneycontrol())
    articles.extend(scrape_economictimes())
    articles.extend(scrape_businessstandard())

    docs = [Document(page_content=text) for text in articles]

    # Split into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    return split_docs

# -------------------------
# Embedding + FAISS caching
# -------------------------
embeddings = OllamaEmbeddings(model="nomic-embed-text")   # ✅ define before cache

@st.cache_resource
def load_vectorstore():
    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        docs = scrape_and_prepare_docs()
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local("faiss_index")
        return vectorstore

# -------------------------
# Streamlit App
# -------------------------
st.set_page_config(page_title="News Research Tool", layout="wide")
st.title("📰 AI Equity News Research Tool")

# Load Vectorstore once
vectorstore = load_vectorstore()

if os.path.exists("faiss_index"):
    st.toast("✅ Using cached FAISS index", icon="🗂️")
else:
    st.toast("📦 FAISS index created & saved")

# Create retriever + QA
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = OllamaLLM(model="mistral")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# User Input
query = st.text_input("🔍 Ask about today's business & economy news:")

if query:
    with st.spinner("Thinking... 🤔"):
        result = qa_chain.invoke(query)
        answer = result["result"]

    st.write("### ✅ Answer")
    st.write(answer)
