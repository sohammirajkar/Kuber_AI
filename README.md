# Kuber – AI-Powered Retrieval & Analytics Agent  

Kuber is an experimental AI agent inspired by platforms like BlackRock’s Aladdin, built to answer complex, context-aware queries over structured and unstructured data. It uses **retrieval-augmented generation (RAG)**, a vector database, and an LLM to deliver accurate, grounded answers.  

## Features  

- Context-aware natural language answers.  
- Pluggable vector stores (TinyVectorStore, FAISS, Chroma).  
- Flexible LLM integration (OpenAI, Cohere, local).  
- Simple REST API backend with analytics endpoints.  
- Easy-to-extend modular design.  

## Requirements  

- Python 3.9+  
- An LLM API key (OpenAI or Cohere) if you want live model responses.  

## Installation  

```bash
git clone https://github.com/yourusername/kuber.git
cd kuber
pip install -r requirements.txt
