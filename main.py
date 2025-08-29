import os
import requests
from bs4 import BeautifulSoup
import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# -------------------------
# Scrapers
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

def scrape_investing_commodities():
    url = "https://www.investing.com/news/commodities-news"
    res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(res.text, "html.parser")
    links = [a["href"] for a in soup.select("a") if a.get("href") and "/news/" in a["href"]]
    articles = []
    for link in links[:5]:
        if not link.startswith("http"):
            link = "https://www.investing.com" + link
        try:
            art_res = requests.get(link, headers={"User-Agent": "Mozilla/5.0"})
            art_soup = BeautifulSoup(art_res.text, "html.parser")
            text = " ".join([p.get_text() for p in art_soup.find_all("p")])
            if text:
                articles.append(text)
        except:
            pass
    return articles

def scrape_investing_forex():
    url = "https://www.investing.com/news/forex-news"
    res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(res.text, "html.parser")
    links = [a["href"] for a in soup.select("a") if a.get("href") and "/news/" in a["href"]]
    articles = []
    for link in links[:5]:
        if not link.startswith("http"):
            link = "https://www.investing.com" + link
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
# Combine scrapers
# -------------------------
def scrape_and_prepare_docs():
    articles = []
    articles.extend(scrape_moneycontrol())
    articles.extend(scrape_economictimes())
    articles.extend(scrape_businessstandard())
    articles.extend(scrape_investing_commodities())
    articles.extend(scrape_investing_forex())

    docs = [Document(page_content=text) for text in articles]
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(docs)

# -------------------------
# Embedding + FAISS
# -------------------------
embeddings = OllamaEmbeddings(model="nomic-embed-text")

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
st.set_page_config(page_title="AI Financial News Tool", layout="wide")
st.title("💹 AI Financial News Research Tool")
st.caption("Equities 📈 | Commodities 🛢️ | Forex 💱")

vectorstore = load_vectorstore()

if os.path.exists("faiss_index"):
    st.toast("✅ Using cached FAISS index", icon="🗂️")
else:
    st.toast("📦 FAISS index created & saved")

# -------------------------
# Custom Prompt (updated for predictions)
# -------------------------
prompt_template = """
You are a financial research assistant.  
Use the following context (today's scraped news) AND your financial reasoning to answer.  

Always structure like this:
📌 **Headline**: One short summary sentence  
📊 **Key Insights**: 3-5 bullet points (analysis, implications, data, risks, opportunities)  
📅 **Sources**: Cite scraped sources (if available). If predicting the future, clearly say it's an AI forecast.  

If the question is about the future, combine news context + reasoning + global macro/market logic.  
If no context is found, say so but still provide thoughtful analysis.  

Context:
{context}

Question: {question}
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = OllamaLLM(model="mistral")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT},
)

# -------------------------
# User Input
# -------------------------
query = st.text_input("🔍 Ask about today's or future markets (Equity, Commodities, Forex):")

if query:
    with st.spinner("Thinking... 🤔"):
        result = qa_chain.invoke(query)
        answer = result["result"]

    st.write("### ✅ AI Market Brief")
    st.write(answer)
