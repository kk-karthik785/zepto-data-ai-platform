import os
import json
from pathlib import Path
from typing import TypedDict, List

import chromadb
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from langgraph.graph import StateGraph, END


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "zepto_policies"

# MOCK_LLM is ON unless MOCK_LLM=0 is explicitly set
MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"


# ============================================================
# EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# CHROMADB
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)


# ============================================================
# LOAD DOCUMENTS
# ============================================================

def load_documents():

    documents = []
    ids = []
    metadatas = []

    for file_path in sorted(DOCS_DIR.glob("doc_*.txt")):

        text = file_path.read_text(
            encoding="utf-8"
        ).strip()

        if text:

            documents.append(text)

            ids.append(file_path.stem)

            metadatas.append({
                "source": file_path.name
            })

    return documents, ids, metadatas


# ============================================================
# BUILD CHROMADB INDEX
# ============================================================

def build_index():

    documents, ids, metadatas = load_documents()

    if len(documents) != 8:

        raise RuntimeError(
            f"Expected 8 documents, but found {len(documents)}."
        )

    embeddings = embedding_model.encode(
        documents,
        normalize_embeddings=True
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(
        f"Successfully indexed {len(documents)} documents."
    )


build_index()


# ============================================================
# STRUCTURED PROMPT
# ============================================================

PROMPT_TEMPLATE = """
ROLE:
You are a Zepto customer support assistant.

CONTEXT:
Use only the Zepto policy information provided in the retrieved context.

TASK:
Answer the customer's question using only the retrieved context.

FORMAT:
Return valid JSON with exactly these fields:
answer, sources, confidence.

LENGTH:
Keep the answer concise and suitable for customer support.

NEGATIVE CONSTRAINT:
Do not use information that is not present in the provided context.
Do not invent or assume Zepto policies.

FEW-SHOT EXAMPLE:

Customer question:
What is the return period for grocery items?

Retrieved context:
Grocery and perishable items may be reported for a return
within 24 hours of delivery if damaged, spoiled, or incorrect.

Example JSON:
{
  "answer": "Grocery and perishable items can be reported for a return within 24 hours of delivery if they are damaged, spoiled, or incorrect.",
  "sources": ["doc_02"],
  "confidence": 1.0
}

RETRIEVED CONTEXT:
{context}

CUSTOMER QUESTION:
{query}
"""


# ============================================================
# PYDANTIC RESPONSE
# ============================================================

class AnswerResponse(BaseModel):

    answer: str

    sources: List[str]

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )


# ============================================================
# FASTAPI REQUEST MODEL
# ============================================================

class AskRequest(BaseModel):

    query: str


# ============================================================
# LANGGRAPH STATE
# ============================================================

class GraphState(TypedDict, total=False):

    query: str

    intent: str

    answer: str

    sources: List[str]

    confidence: float

    retrieved_context: List[str]


# ============================================================
# POLICY KEYWORDS
# ============================================================

POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours"
]


# ============================================================
# NODE 1: CLASSIFY INTENT
# ============================================================

def classify_intent(
    state: GraphState
) -> GraphState:

    query = state["query"]

    lower_query = query.lower()

    # --------------------------------------------------------
    # REQUIRED MOCK MODE
    # --------------------------------------------------------

    if MOCK_LLM:

        if any(
            keyword in lower_query
            for keyword in POLICY_KEYWORDS
        ):

            intent = "policy_question"

        else:

            intent = "general_question"

    # --------------------------------------------------------
    # OPTIONAL REAL LLM MODE
    # --------------------------------------------------------

    else:

        try:

            from langchain_groq import ChatGroq

            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0
            )

            prompt = f"""
Classify the following query as exactly one of:

policy_question
general_question

Use policy_question for questions about:
delivery, returns, refunds, membership,
tracking, cancellation, gift cards,
or support hours.

Query:
{query}

Return only:
policy_question
or
general_question
"""

            result = llm.invoke(prompt)

            classification = (
                result.content
                .strip()
                .lower()
            )

            if classification == "policy_question":

                intent = "policy_question"

            else:

                intent = "general_question"

        except Exception:

            intent = "general_question"

    return {
        **state,
        "intent": intent
    }


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_documents(query: str):

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    ).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    ids = results.get(
        "ids",
        [[]]
    )[0]

    return documents, ids


# ============================================================
# OPTIONAL REAL LLM
# ============================================================

def generate_real_llm_response(
    prompt: str
) -> AnswerResponse:

    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    last_error = ""

    for attempt in range(3):

        if attempt == 0:

            current_prompt = prompt

        else:

            current_prompt = f"""
Your previous response failed validation.

Return ONLY valid JSON.

Required fields:

answer: string
sources: list of strings
confidence: number from 0 to 1

Do not use Markdown.
Do not add explanations.

Original prompt:
{prompt}

Validation error:
{last_error}
"""

        try:

            result = llm.invoke(
                current_prompt
            )

            raw_text = (
                result.content
                .strip()
            )

            data = json.loads(
                raw_text
            )

            validated = AnswerResponse(
                **data
            )

            return validated

        except Exception as error:

            last_error = str(error)

    return AnswerResponse(
        answer=(
            "ERROR: The real LLM response "
            "failed validation after 3 attempts."
        ),
        sources=[],
        confidence=0.0
    )


# ============================================================
# NODE 2: RETRIEVE AND ANSWER
# ============================================================

def retrieve_and_answer(
    state: GraphState
) -> GraphState:

    query = state["query"]

    documents, ids = retrieve_documents(
        query
    )

    if not documents:

        return {
            **state,
            "answer": (
                "No relevant Zepto policy "
                "information was found."
            ),
            "sources": [],
            "confidence": 0.0,
            "retrieved_context": []
        }

    # --------------------------------------------------------
    # REQUIRED MOCK MODE
    # --------------------------------------------------------

    if MOCK_LLM:

        top_chunk_snippet = (
            documents[0][:200]
        )

        answer = (
            "Based on the retrieved context: "
            + top_chunk_snippet
        )

        confidence = 1.0

        sources = ids

    # --------------------------------------------------------
    # OPTIONAL REAL LLM MODE
    # --------------------------------------------------------

    else:

        context = "\n\n".join(
            documents
        )

        prompt = PROMPT_TEMPLATE.format(
            context=context,
            query=query
        )

        llm_response = (
            generate_real_llm_response(
                prompt
            )
        )

        answer = llm_response.answer

        sources = ids

        confidence = (
            llm_response.confidence
        )

    return {
        **state,
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "retrieved_context": documents
    }


# ============================================================
# NODE 3: DIRECT ANSWER
# ============================================================

def direct_answer(
    state: GraphState
) -> GraphState:

    query = state["query"]

    # --------------------------------------------------------
    # REQUIRED MOCK MODE
    # --------------------------------------------------------

    if MOCK_LLM:

        answer = (
            "I can only answer questions about "
            "Zepto policies right now."
        )

        sources = []

        confidence = 1.0

    # --------------------------------------------------------
    # OPTIONAL REAL LLM MODE
    # --------------------------------------------------------

    else:

        prompt = f"""
ROLE:
You are a Zepto customer support assistant.

TASK:
Answer the following general question briefly.

QUESTION:
{query}

FORMAT:
Return valid JSON with:
answer, sources, confidence.

NEGATIVE CONSTRAINT:
Do not invent Zepto policies.
"""

        llm_response = (
            generate_real_llm_response(
                prompt
            )
        )

        answer = llm_response.answer

        sources = []

        confidence = (
            llm_response.confidence
        )

    return {
        **state,
        "answer": answer,
        "sources": sources,
        "confidence": confidence
    }


# ============================================================
# CONDITIONAL ROUTER
# ============================================================

def route_intent(
    state: GraphState
):

    if state["intent"] == "policy_question":

        return "retrieve_and_answer"

    return "direct_answer"


# ============================================================
# BUILD LANGGRAPH
# ============================================================

graph_builder = StateGraph(
    GraphState
)

graph_builder.add_node(
    "classify_intent",
    classify_intent
)

graph_builder.add_node(
    "retrieve_and_answer",
    retrieve_and_answer
)

graph_builder.add_node(
    "direct_answer",
    direct_answer
)

graph_builder.set_entry_point(
    "classify_intent"
)

graph_builder.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "retrieve_and_answer":
            "retrieve_and_answer",

        "direct_answer":
            "direct_answer"
    }
)

graph_builder.add_edge(
    "retrieve_and_answer",
    END
)

graph_builder.add_edge(
    "direct_answer",
    END
)

graph = graph_builder.compile()


# ============================================================
# ASK FUNCTION
# ============================================================

def ask_question(
    query: str
) -> AnswerResponse:

    if not query.strip():

        raise ValueError(
            "Query cannot be empty."
        )

    initial_state: GraphState = {
        "query": query
    }

    result = graph.invoke(
        initial_state
    )

    response = AnswerResponse(
        answer=result["answer"],
        sources=result.get(
            "sources",
            []
        ),
        confidence=result.get(
            "confidence",
            0.0
        )
    )

    return response


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Zepto Support Assistant",
    description=(
        "Offline RAG-based Zepto "
        "policy support assistant"
    ),
    version="1.0.0"
)


@app.get("/")
def root():

    return {
        "message":
            "Zepto Support Assistant is running",

        "mock_llm":
            MOCK_LLM
    }


@app.post(
    "/ask",
    response_model=AnswerResponse
)
def ask(
    request: AskRequest
):

    try:

        return ask_question(
            request.query
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("========================================")
    print("Zepto Support Assistant")
    print("========================================")

    print(
        f"MOCK_LLM: {MOCK_LLM}"
    )

    print()
    print("Policy question example:")

    result1 = ask_question(
        "What is the delivery fee for orders below INR 149?"
    )

    print(
        result1.model_dump_json(
            indent=2
        )
    )

    print()
    print("General question example:")

    result2 = ask_question(
        "What is the capital of India?"
    )

    print(
        result2.model_dump_json(
            indent=2
        )
    )