# AI Equity News Research Tool

This is a Streamlit-based AI tool that scrapes equity/business news from multiple sources,
stores them in a FAISS vector store, and answers user queries using LangChain + Ollama LLM.

## Features
- Scrapes news from MoneyControl, Economic Times, Business Standard
- Embeds articles with Ollama embeddings
- Uses RetrievalQA to answer questions
- Streamlit UI for easy interaction

## How to Run
```bash
pip install -r requirements.txt
streamlit run main.py
