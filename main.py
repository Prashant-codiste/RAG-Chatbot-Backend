import os
import pyarrow as pa
import lancedb
import requests
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from langchain_openai import ChatOpenAI

load_dotenv()

# Cache & config directories
os.environ.setdefault("TRANSFORMERS_CACHE", "./models_cache")
os.environ.setdefault("HF_HOME", "./models_cache")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "./models_cache/sentence_transformers")
os.makedirs("./models_cache", exist_ok=True)
os.makedirs("./models_cache/sentence_transformers", exist_ok=True)

# Configuration constants
LANCEDB_PATH = "./lancedb_data"
TABLE_NAME_CN = "cn_dump_embeddings"
TABLE_NAME_DOC = "document_embeddings"
API_BASE_URL = "http://103.211.171.130/eMistClientTPI/api/TPSBestAI/GetCNDump8006"
API_AUTH_TOKEN = "Sync SU5UTDpCRVNUOjIwMjUtMDktMTU="
DEFAULT_DAYS_BACK = 30

# Globals
vectorstore: Optional["LanceDBVectorStore"] = None
qa_chain = None
_db = None
embeddings = None
llm = None

# LanceDB schema for embeddings
schema = pa.schema([
    pa.field("vector", pa.list_(pa.float32(), 384)),
    pa.field("text", pa.utf8()),
    pa.field("source", pa.utf8()),
    pa.field("chunk_id", pa.int32()),
])


class LanceDBVectorStore(VectorStore):
    def __init__(self, tables, embedding_function):
        self.tables = tables if isinstance(tables, list) else [tables]
        self.embedding_function = embedding_function

    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None, **kwargs) -> List[str]:
        embeddings_list = self.embedding_function.embed_documents(texts)
        data = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings_list)):
            metadata = metadatas[i] if metadatas else {}
            data.append({
                "vector": embedding,
                "text": text,
                "source": metadata.get("source", "CN_API"),
                "chunk_id": metadata.get("chunk_id", i)
            })
        for table in self.tables:
            table.add(data)
        return [str(i) for i in range(len(texts))]

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[Document]:
        query_embedding = self.embedding_function.embed_query(query)
        all_docs = []
        for table in self.tables:
            try:
                results = table.search(query_embedding).limit(k).to_pandas()
                for _, row in results.iterrows():
                    sim = row.get("_distance", None)
                    doc = Document(
                        page_content=row["text"],
                        metadata={
                            "source": row.get("source", "CN_API"),
                            "chunk_id": int(row.get("chunk_id", 0)),
                            "similarity_score": float(sim) if sim is not None else None,
                            "embedding_dimensions": 384
                        }
                    )
                    score = float(sim) if sim is not None else float("inf")
                    all_docs.append((score, doc))
            except Exception as e:
                print("Search error on table:", e)
                continue

        all_docs.sort(key=lambda x: x[0])
        return [doc for _, doc in all_docs[:k]]

    def _get_relevant_documents(self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None) -> List[Document]:
        return self.similarity_search(query, k=4)

    def as_retriever(self, **kwargs):
        from langchain_core.retrievers import BaseRetriever

        class SimpleRetriever(BaseRetriever):
            vectorstore: LanceDBVectorStore
            search_kwargs: dict = {}

            class Config:
                arbitrary_types_allowed = True

            def _get_relevant_documents(self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None) -> List[Document]:
                k = self.search_kwargs.get("k", 3)
                return self.vectorstore.similarity_search(query, k=k)

        search_kwargs = kwargs.get("search_kwargs", {})
        return SimpleRetriever(vectorstore=self, search_kwargs=search_kwargs)

    @classmethod
    def from_texts(cls, texts: List[str], embedding, metadatas: Optional[List[dict]] = None, **kwargs):
        raise NotImplementedError("from_texts not supported; use existing LanceDB tables.")


def fetch_cn_data(from_date: Optional[str] = None, to_date: Optional[str] = None) -> List[Dict[str, Any]]:
    if not from_date:
        from_date = (datetime.now() - timedelta(days=DEFAULT_DAYS_BACK)).strftime("%Y-%m-%d")
    if not to_date:
        to_date = datetime.now().strftime("%Y-%m-%d")

    headers = {"Authorization": API_AUTH_TOKEN, "Content-Type": "application/json"}
    params = {"FromDate": from_date, "Todate": to_date}

    response = requests.get(API_BASE_URL, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def process_cn_data(cn_records: List[Dict[str, Any]]) -> List[str]:
    docs = []
    for idx, rec in enumerate(cn_records):
        s = f"CN Record {idx+1}:\n"
        for k,v in rec.items():
            if v is not None and v != "":
                s += f"{k}: {v}\n"
        s += "\n"
        docs.append(s)
    return docs


def initialize_chatbot():
    global vectorstore, qa_chain, _db, embeddings, llm

    print("Connecting to LanceDB...")
    _db = lancedb.connect(LANCEDB_PATH)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="./models_cache/sentence_transformers"
    )

    table_names = _db.table_names()
    tables = []

    if TABLE_NAME_DOC in table_names:
        tables.append(_db.open_table(TABLE_NAME_DOC))
        print(f"Loaded {TABLE_NAME_DOC} table")
    else:
        print(f"Table {TABLE_NAME_DOC} not found — fetching CN data & building embeddings...")
        cn = fetch_cn_data()
        if not cn:
            print("No CN data returned from API.")
            return False
        docs = process_cn_data(cn)
        splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
        text_objs = splitter.create_documents(docs)

        cnt = len(text_objs)
        print(f"Embedding {cnt} documents...")
        batch_size = 64
        all_data = []
        texts = [d.page_content for d in text_objs]
        for i in range(0, cnt, batch_size):
            batch = texts[i:i+batch_size]
            embs = embeddings.embed_documents(batch)
            for j, (t, e) in enumerate(zip(batch, embs)):
                all_data.append({"vector": e, "text": t, "source": "CN_API", "chunk_id": i+j})

        tbl = _db.create_table(TABLE_NAME_DOC, data=all_data, schema=schema)
        tables.append(tbl)
        print(f"Created table {TABLE_NAME_DOC} with {len(all_data)} embeddings.")

    print(f"Using {len(tables)} table(s) for vectorstore.")
    vectorstore = LanceDBVectorStore(tables, embeddings)

    print("Initializing ChatOpenAI LLM...")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is not set. Aborting LLM setup.")
        return False

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",        # change to preferred model
        temperature=0.3,
        max_tokens=2048,
        max_retries=2,
    )

    prompt = PromptTemplate(
        template=(
            "You are a data analyst answering based on CN data. Use the context and reply clearly.\n\n"
            "Context: {context}\nQuestion: {question}\n\nAnswer:"
        ),
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    print("Chatbot ready")
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up...")
    success = initialize_chatbot()
    if not success:
        print("Initialization failed.")
    yield
    print("Shutting down.")


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


class Query(BaseModel):
    question: str
    include_vectors: bool = False


class DocumentInfo(BaseModel):
    content: str
    source: str
    chunk_id: int
    similarity_score: Optional[float] = None
    embedding_dimensions: int = 384
    embedding_vector: Optional[List[float]] = None


class ChatResponse(BaseModel):
    answer: str
    source_documents: Optional[List[DocumentInfo]] = None
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    include_vectors: bool = False


@app.post("/query", response_model=ChatResponse)
async def query_chatbot(query: Query):
    if qa_chain is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = qa_chain({"query": query.question})

    docs = []
    for doc in result.get("source_documents", []):
        content = doc.page_content
        if len(content) > 500:
            content = content[:500] + "..."
        chunk_id = int(doc.metadata.get("chunk_id", 0))

        embedding_vector = None
        if query.include_vectors:
            for tbl in vectorstore.tables:
                df = tbl.to_pandas()
                row = df[df["chunk_id"] == chunk_id]
                if not row.empty:
                    vector = row.iloc[0]["vector"]
                    embedding_vector = vector.tolist() if hasattr(vector, "tolist") else list(vector)
                    break

        docs.append(DocumentInfo(
            content=content,
            source=doc.metadata.get("source", "CN_API"),
            chunk_id=chunk_id,
            similarity_score=doc.metadata.get("similarity_score"),
            embedding_dimensions=doc.metadata.get("embedding_dimensions", 384),
            embedding_vector=embedding_vector
        ))

    return ChatResponse(answer=result.get("result", ""), source_documents=docs, include_vectors=query.include_vectors)


@app.get("/health")
def health_check():
    return {
        "status": "healthy" if vectorstore is not None else "initializing",
        "vectorstore_loaded": vectorstore is not None,
        "qa_chain_loaded": qa_chain is not None,
    }


@app.get("/")
def root():
    return {"message": "CN Dump RAG Chatbot API", "status": "ok"}
