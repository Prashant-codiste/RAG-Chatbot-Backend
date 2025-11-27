# RAG Chatbot Backend

FastAPI backend for the CN Dump RAG Chatbot application. This service provides a RAG (Retrieval-Augmented Generation) API for querying CN (Consignment Note) data using OpenAI and LanceDB.

## Features

- **RAG-based Query System**: Query CN data using natural language
- **LanceDB Vector Store**: Efficient similarity search using document embeddings
- **OpenAI Integration**: Uses GPT models for intelligent responses
- **CN API Integration**: Fetches data from external CN Dump API
- **Batch Processing**: Efficient embedding creation in batches

## Tech Stack

- **FastAPI**: Web framework
- **LangChain**: LLM orchestration
- **OpenAI**: Language model (GPT-3.5-turbo / GPT-4o-mini)
- **LanceDB**: Local vector database for embeddings
- **HuggingFace**: Sentence transformers for embeddings
- **PyArrow**: Schema definition for LanceDB

## Setup

### Prerequisites

- Python 3.12+
- pip

### Installation

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo  # Optional: defaults to gpt-3.5-turbo
```

### API Configuration

The CN Dump API configuration is set in `main.py`:
- `API_BASE_URL`: CN Dump API endpoint
- `API_AUTH_TOKEN`: Authentication token
- `DEFAULT_DAYS_BACK`: Default days to fetch data (30)

## Running the Server

```bash
# Activate virtual environment
source venv/bin/activate

# Start the server
uvicorn main:app --port 8002 --host 0.0.0.0
```

The API will be available at:
- **API**: http://localhost:8002
- **Swagger UI**: http://localhost:8002/docs
- **ReDoc**: http://localhost:8002/redoc

## API Endpoints

### `POST /query`
Query the chatbot with a question.

**Request:**
```json
{
  "question": "What is the total number of CN records?",
  "include_vectors": false
}
```

**Response:**
```json
{
  "answer": "...",
  "source_documents": [...],
  "include_vectors": false
}
```

### `GET /health`
Health check endpoint.

### `GET /`
API information and available endpoints.

## Project Structure

```
.
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (not in git)
├── lancedb_data/       # LanceDB vector database (not in git)
├── models_cache/        # Cached models (not in git)
└── README.md           # This file
```

## Data Storage

- **LanceDB**: Local vector database stored in `lancedb_data/`
- **Embeddings**: Uses `document_embeddings` table
- **Model Cache**: Cached models in `models_cache/`

## Development

### Adding New Endpoints

Add new endpoints in `main.py` following FastAPI patterns.

### Updating Embeddings

The system automatically loads existing embeddings on startup. To refresh:
1. Delete `lancedb_data/document_embeddings.lance/`
2. Restart the server (it will recreate from CN API)

## Troubleshooting

### Port Already in Use
Change the port in the uvicorn command:
```bash
uvicorn main:app --port 8003 --host 0.0.0.0
```

### OpenAI API Key Not Found
Ensure `.env` file exists with `OPENAI_API_KEY` set.

### Embeddings Not Loading
Check that `lancedb_data/document_embeddings.lance/` exists and is readable.

## License

[Your License Here]
