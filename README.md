# 💹 AI Financial News Research Tool

A **Streamlit** app that scrapes the latest financial news and provides AI-powered insights for **Equities**, **Commodities**, and **Forex** markets using **LangChain + Ollama LLMs + FAISS**.

---

## Features

- Scrapes financial news from multiple sources:
  - Moneycontrol
  - Economic Times
  - Business Standard
  - Investing (Commodities & Forex)
- Converts scraped articles into embeddings using **OllamaEmbeddings**.
- Stores embeddings locally using **FAISS** for fast retrieval.
- AI-powered question answering based on current financial news.
- Provides structured output:
  - **Headline**: One short summary sentence  
  - **Key Insights**: 3-5 bullet points (analysis, implications, risks, opportunities)  
  - **Sources**: Cites scraped sources if available

---

## Demo

- Local URL: `http://localhost:8501`
- Network URL: `http://<your-local-ip>:8501`

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/RajShah03/AI-Financial-News-Research-Tool.git
cd AI-Financial-News-Research-Tool
```
Install dependencies:

```bash
pip install -r requirements.txt
```
Run the app:

```bash
streamlit run main.py
```

Usage
Open the Streamlit app.

Ask questions about today’s or future markets in the input box:


Example: Given the recent RBI policy updates, ongoing US-China trade tensions, and global crude oil trends, what is the likely impact on India’s top 3 IT and Pharma stocks over the next quarter?
View AI-generated market brief with headline, key insights, and sources.

Tech Stack
Python 3.10+

Streamlit

LangChain

Ollama LLM & Embeddings

FAISS

BeautifulSoup for web scraping

Requests library

Notes
The FAISS index is cached locally for faster responses.

The app is optimized for CPU usage.