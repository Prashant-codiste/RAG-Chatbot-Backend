import os
import logging
import lancedb
import requests
import pyarrow as pa
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.base import VectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.callbacks import CallbackManagerForRetrieverRun

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

os.makedirs("./lancedb_data", exist_ok=True)
LANCEDB_PATH = "./lancedb_data"
TABLE_NAME_DOC = "document_embeddings"
API_BASE_URL = "http://103.211.171.130/eMistClientTPI/api/TPSBestAI/GetCNDump8006"
API_AUTH_TOKEN = "Sync SU5UTDpCRVNUOjIwMjUtMDktMTU="
DEFAULT_DAYS_BACK = 30
EMBEDDING_DIMENSIONS = 1536
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 200
BATCH_SIZE = 64

vectorstore: Optional["LanceDBVectorStore"] = None
qa_chain = None
_db = None
embeddings = None
llm = None

schema = pa.schema([
    pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIMENSIONS)),
    pa.field("text", pa.utf8()),
    pa.field("source", pa.utf8()),
    pa.field("chunk_id", pa.int32()),
])


def validate_environment() -> bool:
    """Validate that all required environment variables are set."""
    required_vars = ["OPENAI_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        return False
    
    logger.info("All required environment variables are set")
    return True


class LanceDBVectorStore(VectorStore):
    """Custom LanceDB vector store implementation."""
    
    def __init__(self, tables, embedding_function):
        self.tables = tables if isinstance(tables, list) else [tables]
        self.embedding_function = embedding_function
        logger.info(f"Initialized LanceDBVectorStore with {len(self.tables)} table(s)")

    def add_texts(
        self, 
        texts: List[str], 
        metadatas: Optional[List[dict]] = None, 
        **kwargs
    ) -> List[str]:
        """Add texts to the vector store."""
        try:
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
            
            logger.info(f"Added {len(texts)} texts to vector store")
            return [str(i) for i in range(len(texts))]
            
        except Exception as e:
            logger.error(f"Error adding texts to vector store: {e}")
            raise

    def similarity_search(
        self, 
        query: str, 
        k: int = 4, 
        **kwargs
    ) -> List[Document]:
        """Perform similarity search using stored document embeddings."""
        try:
            total_document_embeddings = 0
            tables_with_data = []
            
            for table in self.tables:
                try:
                    df = table.to_pandas()
                    count = len(df)
                    if count > 0:
                        total_document_embeddings += count
                        tables_with_data.append((table, count))
                        logger.debug(f"Table has {count} document embeddings")
                except Exception as e:
                    logger.warning(f"Error checking table count: {e}")
                    continue
            
            if total_document_embeddings == 0:
                logger.warning("No document embeddings found in local storage - cannot perform search")
                return []
            
            logger.info(f"Searching against {total_document_embeddings} stored document embeddings")
            query_embedding = self.embedding_function.embed_query(query)
            logger.debug(f"Generated query embedding (dimension: {len(query_embedding)})")
            
            all_docs = []
            
            for table, doc_count in tables_with_data:
                try:
                    logger.debug(f"Searching table with {doc_count} document embeddings")
                    
                    df = table.to_pandas()
                    if len(df) > 0 and "vector" in df.columns:
                        stored_dim = len(df.iloc[0]["vector"]) if hasattr(df.iloc[0]["vector"], "__len__") else None
                        query_dim = len(query_embedding)
                        
                        if stored_dim and stored_dim != query_dim:
                            logger.error(f"Dimension mismatch: stored embeddings have {stored_dim} dimensions, but query has {query_dim} dimensions")
                            logger.warning("Embeddings need to be regenerated with correct dimensions")
                            raise ValueError(f"Embedding dimension mismatch: {stored_dim} != {query_dim}")
                    
                    results = table.search(query_embedding).limit(k).to_pandas()
                    logger.debug(f"Found {len(results)} matches from this table")
                    
                    for _, row in results.iterrows():
                        sim = row.get("_distance", None)
                        doc = Document(
                            page_content=row["text"],
                            metadata={
                                "source": row.get("source", "CN_API"),
                                "chunk_id": int(row.get("chunk_id", 0)),
                                "similarity_score": float(sim) if sim is not None else None,
                                "embedding_dimensions": EMBEDDING_DIMENSIONS
                            }
                        )
                        score = float(sim) if sim is not None else float("inf")
                        all_docs.append((score, doc))
                        
                except ValueError as e:
                    if "dimension mismatch" in str(e).lower() or "size" in str(e).lower():
                        logger.error(f"Embedding dimension mismatch detected: {e}")
                        raise
                    logger.warning(f"Search error on table with {doc_count} embeddings: {e}")
                    continue
                except Exception as e:
                    error_msg = str(e).lower()
                    if "size" in error_msg and "match" in error_msg:
                        logger.error(f"Embedding dimension mismatch: {e}")
                        raise ValueError(f"Embedding dimension mismatch detected: {e}")
                    logger.warning(f"Search error on table with {doc_count} embeddings: {e}")
                    continue

            all_docs.sort(key=lambda x: x[0])
            results = [doc for _, doc in all_docs[:k]]
            logger.info(f"Found {len(results)} relevant documents using stored document embeddings")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search using document embeddings: {e}")
            raise

    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """Get relevant documents for a query."""
        return self.similarity_search(query, k=4)

    def as_retriever(self, **kwargs):
        """Return a retriever interface for this vector store."""
        from langchain_core.retrievers import BaseRetriever

        class SimpleRetriever(BaseRetriever):
            vectorstore: LanceDBVectorStore
            search_kwargs: dict = {}

            class Config:
                arbitrary_types_allowed = True

            def _get_relevant_documents(
                self, 
                query: str, 
                *, 
                run_manager: Optional[CallbackManagerForRetrieverRun] = None
            ) -> List[Document]:
                k = self.search_kwargs.get("k", 3)
                return self.vectorstore.similarity_search(query, k=k)

        search_kwargs = kwargs.get("search_kwargs", {})
        return SimpleRetriever(vectorstore=self, search_kwargs=search_kwargs)

    @classmethod
    def from_texts(
        cls, 
        texts: List[str], 
        embedding, 
        metadatas: Optional[List[dict]] = None, 
        **kwargs
    ):
        """Not implemented - use existing LanceDB tables."""
        raise NotImplementedError("from_texts not supported; use existing LanceDB tables.")


def fetch_cn_data(
    from_date: Optional[str] = None, 
    to_date: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Fetch CN data from the API."""
    if not from_date:
        from_date = (datetime.now() - timedelta(days=DEFAULT_DAYS_BACK)).strftime("%Y-%m-%d")
    if not to_date:
        to_date = datetime.now().strftime("%Y-%m-%d")

    headers = {
        "Authorization": API_AUTH_TOKEN, 
        "Content-Type": "application/json"
    }
    params = {
        "FromDate": from_date, 
        "Todate": to_date
    }

    try:
        logger.info(f"Fetching CN data from {from_date} to {to_date}")
        response = requests.get(
            API_BASE_URL, 
            headers=headers, 
            params=params, 
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        logger.info(f"Successfully fetched {len(data)} CN records")
        return data
        
    except requests.exceptions.Timeout:
        logger.error("API request timed out")
        raise HTTPException(
            status_code=504, 
            detail="API request timed out"
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching CN data: {e}")
        raise HTTPException(
            status_code=502, 
            detail=f"Failed to fetch CN data: {str(e)}"
        )
    except ValueError as e:
        logger.error(f"Error parsing JSON response: {e}")
        raise HTTPException(
            status_code=502, 
            detail="Invalid JSON response from API"
        )


def process_cn_data(cn_records: List[Dict[str, Any]]) -> List[str]:
    """Process CN records into text documents."""
    docs = []
    
    for idx, rec in enumerate(cn_records):
        s = f"CN Record {idx+1}:\n"
        for k, v in rec.items():
            if v is not None and v != "":
                s += f"{k}: {v}\n"
        s += "\n"
        docs.append(s)
    
    logger.info(f"Processed {len(docs)} CN records into documents")
    return docs


def create_embeddings_table(db, docs: List[str], embeddings) -> Any:
    """Create embeddings table from documents."""
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
        text_objs = splitter.create_documents(docs)
        cnt = len(text_objs)
        logger.info(f"Split documents into {cnt} chunks")

        logger.info(f"Creating document embeddings for {cnt} document chunks...")
        all_data = []
        texts = [d.page_content for d in text_objs]
        
        for i in range(0, cnt, BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]
            batch_num = i//BATCH_SIZE + 1
            total_batches = (cnt-1)//BATCH_SIZE + 1
            logger.info(f"Creating document embeddings - batch {batch_num}/{total_batches} ({len(batch)} documents)")
            
            try:
                embs = embeddings.embed_documents(batch)
                logger.debug(f"Generated {len(embs)} document embeddings for batch {batch_num}")
                
                for j, (t, e) in enumerate(zip(batch, embs)):
                    all_data.append({
                        "vector": e,
                        "text": t, 
                        "source": "CN_API", 
                        "chunk_id": i+j
                    })
            except Exception as e:
                logger.error(f"Error creating document embeddings for batch {batch_num}: {e}")
                raise

        tbl = db.create_table(TABLE_NAME_DOC, data=all_data, schema=schema)
        logger.info(f"Stored {len(all_data)} document embeddings in table {TABLE_NAME_DOC}")
        logger.info("Document embeddings are now available for similarity search")
        return tbl
        
    except Exception as e:
        logger.error(f"Error creating embeddings table: {e}")
        raise


def initialize_chatbot() -> bool:
    """Initialize the chatbot with vector store and QA chain."""
    global vectorstore, qa_chain, _db, embeddings, llm

    try:
        # Validate environment
        if not validate_environment():
            return False

        logger.info("Connecting to LanceDB...")
        _db = lancedb.connect(LANCEDB_PATH)

        logger.info("Initializing OpenAI embeddings...")
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002"
        )

        table_names = _db.table_names()
        tables = []

        if TABLE_NAME_DOC in table_names:
            table = _db.open_table(TABLE_NAME_DOC)
            try:
                df = table.to_pandas()
                count = len(df)
                if count > 0:
                    stored_dim = None
                    if "vector" in df.columns and len(df) > 0:
                        first_vector = df.iloc[0]["vector"]
                        if hasattr(first_vector, "__len__"):
                            stored_dim = len(first_vector)
                    
                    test_embedding = embeddings.embed_query("test")
                    current_dim = len(test_embedding)
                    
                    if stored_dim and stored_dim != current_dim:
                        logger.warning(f"Dimension mismatch detected: stored embeddings have {stored_dim} dimensions, but current model produces {current_dim} dimensions")
                        logger.info("Dropping old table and regenerating embeddings with correct dimensions...")
                        _db.drop_table(TABLE_NAME_DOC)
                        cn = fetch_cn_data()
                        if not cn:
                            logger.error("No CN data returned from API")
                            return False
                        
                        docs = process_cn_data(cn)
                        tbl = create_embeddings_table(_db, docs, embeddings)
                        tables.append(tbl)
                    else:
                        tables.append(table)
                        logger.info(f"Loaded existing {TABLE_NAME_DOC} table with {count} embeddings (dimension: {stored_dim or current_dim})")
                else:
                    logger.warning(f"Table {TABLE_NAME_DOC} exists but is empty - will fetch new data")
                    try:
                        cn = fetch_cn_data()
                        if not cn:
                            logger.error("No CN data returned from API")
                            return False
                        
                        docs = process_cn_data(cn)
                        _db.drop_table(TABLE_NAME_DOC)
                        tbl = create_embeddings_table(_db, docs, embeddings)
                        tables.append(tbl)
                    except Exception as e:
                        logger.error(f"Error during data fetch and embedding: {e}")
                        return False
            except Exception as e:
                logger.warning(f"Error checking table: {e}, will regenerate embeddings")
                try:
                    _db.drop_table(TABLE_NAME_DOC)
                    cn = fetch_cn_data()
                    if not cn:
                        logger.error("No CN data returned from API")
                        return False
                    
                    docs = process_cn_data(cn)
                    tbl = create_embeddings_table(_db, docs, embeddings)
                    tables.append(tbl)
                except Exception as e2:
                    logger.error(f"Error during data fetch and embedding: {e2}")
                    return False
        else:
            logger.info(f"Table {TABLE_NAME_DOC} not found - fetching CN data & building embeddings...")
            
            try:
                cn = fetch_cn_data()
                if not cn:
                    logger.error("No CN data returned from API")
                    return False
                
                docs = process_cn_data(cn)
                tbl = create_embeddings_table(_db, docs, embeddings)
                tables.append(tbl)
                
            except Exception as e:
                logger.error(f"Error during initial data fetch and embedding: {e}")
                return False

        logger.info(f"Using {len(tables)} table(s) for vectorstore")
        vectorstore = LanceDBVectorStore(tables, embeddings)

        logger.info("Initializing ChatOpenAI LLM...")
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=2048,
            max_retries=2
        )

        prompt = PromptTemplate(
            template=(
                "You are a helpful data analyst assistant answering questions based on CN (Credit Note) data.\n"
                "Use the provided context to answer the question clearly and concisely.\n"
                "If the answer is not in the context, say so - do not make up information.\n\n"
                "Context: {context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            ),
            input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

        logger.info("Chatbot initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing chatbot: {e}", exc_info=True)
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    logger.info("Starting up application...")
    success = initialize_chatbot()
    
    if not success:
        logger.error("Failed to initialize chatbot")
    else:
        logger.info("Application startup complete")
    
    yield
    
    logger.info("Shutting down application...")


app = FastAPI(
    title="CN Dump RAG Chatbot API",
    description="RAG-based chatbot for querying CN (Credit Note) data",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class Query(BaseModel):
    question: str
    include_vectors: bool = False


class DocumentInfo(BaseModel):
    content: str
    source: str
    chunk_id: int
    similarity_score: Optional[float] = None
    embedding_dimensions: int = EMBEDDING_DIMENSIONS
    embedding_vector: Optional[List[float]] = None


class ChatResponse(BaseModel):
    answer: str
    source_documents: Optional[List[DocumentInfo]] = None
    embedding_model: str = "text-embedding-ada-002"
    include_vectors: bool = False


class HealthResponse(BaseModel):
    status: str
    vectorstore_loaded: bool
    qa_chain_loaded: bool
    lancedb_connected: bool


@app.post("/query", response_model=ChatResponse)
async def query_chatbot(query: Query):
    """Query the chatbot with a question."""
    global vectorstore, _db, embeddings, qa_chain, llm
    
    if qa_chain is None:
        logger.error("Chatbot not initialized - received query request")
        raise HTTPException(
            status_code=503, 
            detail="Chatbot not initialized. Please try again later."
        )
    
    if not query.question.strip():
        raise HTTPException(
            status_code=400, 
            detail="Question cannot be empty"
        )
    
    try:
        logger.info(f"Processing query: {query.question[:100]}...")
        try:
            result = qa_chain({"query": query.question})
        except ValueError as e:
            error_msg = str(e).lower()
            if "dimension" in error_msg and "mismatch" in error_msg:
                logger.error("Embedding dimension mismatch detected during query")
                logger.info("Attempting to regenerate embeddings...")
                
                try:
                    if _db and TABLE_NAME_DOC in _db.table_names():
                        _db.drop_table(TABLE_NAME_DOC)
                        logger.info("Dropped old embeddings table")
                    
                    cn = fetch_cn_data()
                    if not cn:
                        raise HTTPException(status_code=503, detail="Failed to fetch CN data for embedding regeneration")
                    
                    docs = process_cn_data(cn)
                    tbl = create_embeddings_table(_db, docs, embeddings)
                    
                    vectorstore = LanceDBVectorStore([tbl], embeddings)
                    
                    prompt = PromptTemplate(
                        template=(
                            "You are a helpful data analyst assistant answering questions based on CN (Credit Note) data.\n"
                            "Use the provided context to answer the question clearly and concisely.\n"
                            "If the answer is not in the context, say so - do not make up information.\n\n"
                            "Context: {context}\n\n"
                            "Question: {question}\n\n"
                            "Answer:"
                        ),
                        input_variables=["context", "question"]
                    )
                    
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
                        chain_type_kwargs={"prompt": prompt},
                        return_source_documents=True
                    )
                    
                    logger.info("Embeddings regenerated successfully, retrying query...")
                    result = qa_chain({"query": query.question})
                except Exception as regen_error:
                    logger.error(f"Failed to regenerate embeddings: {regen_error}")
                    raise HTTPException(
                        status_code=503,
                        detail="Embedding dimension mismatch. Please restart the server to regenerate embeddings."
                    )
            else:
                raise

        docs = []
        for doc in result.get("source_documents", []):
            content = doc.page_content
            if len(content) > 500:
                content = content[:500] + "..."
            
            chunk_id = int(doc.metadata.get("chunk_id", 0))

            embedding_vector = None
            if query.include_vectors:
                try:
                    for tbl in vectorstore.tables:
                        df = tbl.to_pandas()
                        row = df[df["chunk_id"] == chunk_id]
                        if not row.empty:
                            vector = row.iloc[0]["vector"]
                            embedding_vector = vector.tolist() if hasattr(vector, "tolist") else list(vector)
                            break
                except Exception as e:
                    logger.warning(f"Error retrieving embedding vector: {e}")

            docs.append(DocumentInfo(
                content=content,
                source=doc.metadata.get("source", "CN_API"),
                chunk_id=chunk_id,
                similarity_score=doc.metadata.get("similarity_score"),
                embedding_dimensions=doc.metadata.get("embedding_dimensions", EMBEDDING_DIMENSIONS),
                embedding_vector=embedding_vector
            ))

        answer = result.get("result", "")
        logger.info(f"Query processed successfully. Answer length: {len(answer)}")
        
        return ChatResponse(
            answer=answer,
            source_documents=docs,
            include_vectors=query.include_vectors
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    is_healthy = vectorstore is not None and qa_chain is not None
    
    return HealthResponse(
        status="healthy" if is_healthy else "initializing",
        vectorstore_loaded=vectorstore is not None,
        qa_chain_loaded=qa_chain is not None,
        lancedb_connected=_db is not None
    )


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "CN Dump RAG Chatbot API",
        "status": "ok",
        "version": "1.0.0",
        "endpoints": {
            "query": "/query",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/stats")
def get_stats():
    """Get statistics about the vector store."""
    if vectorstore is None or _db is None:
        raise HTTPException(
            status_code=503,
            detail="Vector store not initialized"
        )
    
    try:
        stats = {
            "tables": [],
            "total_embeddings": 0
        }
        
        for table in vectorstore.tables:
            try:
                table_name = table.name
                df = table.to_pandas()
                count = len(df)
                stats["tables"].append({
                    "name": table_name,
                    "row_count": count
                })
                stats["total_embeddings"] += count
            except Exception as e:
                logger.warning(f"Error getting stats for table: {e}")
                continue
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting stats: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
