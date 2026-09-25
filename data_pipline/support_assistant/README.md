# Zepto Support Assistant

## Overview

This module implements a local Zepto policy support assistant using document
embeddings, ChromaDB retrieval, LangGraph, Pydantic validation, and FastAPI.

The default graded mode uses `MOCK_LLM=True`, so no LLM API key is required.

## Architecture

```text
8 Zepto policy documents
        |
        v
Document ingestion
        |
        v
SentenceTransformer
all-MiniLM-L6-v2
        |
        v
ChromaDB
Cosine similarity
        |
        v
LangGraph
        |
   +----+----+
   |         |
Policy     General
question   question
   |         |
   v         v
Retrieve   Direct
top-3      answer
   |
   v
Pydantic response
   |
   v
FastAPI /ask