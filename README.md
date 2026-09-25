\# Zepto Data \& AI Platform



\## Project Overview



This project contains three modules developed for the Zepto AI/ML capstone:



1\. Data Pipeline – Web scraping, data cleaning, SQLite database, and SQL analysis.

2\. Analytics – Titanic dataset EDA, classification models, imbalance handling, regression analysis, and model comparison.

3\. Support Assistant – Local RAG-based Zepto policy assistant using Sentence Transformers, ChromaDB, LangGraph, Pydantic, and FastAPI.



\## Project Structure



zepto-data-ai-platform/

├── data\_pipline/

│   ├── pipline.py

│   ├── books.db

│   ├── books\_cleaned.csv

│   ├── queries.sql

│   ├── query\_outputs.txt

│   └── support\_assistant/

│       ├── main.py

│       ├── requirements.txt

│       ├── Dockerfile

│       ├── README.md

│       ├── docs/

│       │   ├── doc\_01.txt

│       │   ├── doc\_02.txt

│       │   ├── doc\_03.txt

│       │   ├── doc\_04.txt

│       │   ├── doc\_05.txt

│       │   ├── doc\_06.txt

│       │   ├── doc\_07.txt

│       │   └── doc\_08.txt

│       └── chroma\_db/

├── analytics/

│   └── titanic\_analysis.py

├── outputs/

├── titanic.csv

└── README.md



\## Module 1 – Data Pipeline



Source: Books to Scrape



The pipeline:

\- Scrapes 100 books.

\- Extracts title, price, rating, availability, and category.

\- Cleans the scraped data.

\- Converts GBP to INR using 1 GBP = 105.50 INR.

\- Stores normalized data in SQLite.

\- Executes SQL queries using filtering, sorting, limiting, distinct values, ranges, and joins.

\- Loads SQL results into pandas

